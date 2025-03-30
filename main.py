import time
import uuid
import requests
import argparse
import functools

import gradio as gr
import folium

from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.prompts import PromptTemplateManager

class Agent:
    def __init__(self, credentials, project_id, model_id, prompt, params):
        self.model = ModelInference(model_id=model_id,
                                    credentials=credentials,
                                    project_id=project_id)

        self.prompt = prompt
        self.params = params

    def get_params(self):
        return self.params

    def get_prompt(self):
        return self.prompt

    def generate(self, prompt, params=None):
        return self.model.generate_text(prompt, params=params if params is not None else self.params, guardrails=True)

def get_long_lat_from_postcode(postcode):
    response = requests.get(f"https://api.postcodes.io/postcodes/{postcode}")
    result = response.json()["result"]
    return result["longitude"], result["latitude"]

def station_to_text(x):
    label, river_name, long, lat = x
    return f"label: {label}, river_name: {river_name}, long: {long}, lat: {lat}"

def measure_to_text(x):
    parameter, value, unit = x
    return f"{parameter}: {value} {unit}"

def get_hydrology_info(long, lat, dist=4):
    url = f"https://environment.data.gov.uk/hydrology/id/stations?lat={lat}&long={long}&dist={dist}"
    suffix = "/readings?latest"

    response = requests.get(url)

    stations = []
    measures = []
    text = ""

    for i in response.json()["items"]:
        label = i["label"]
        river_name = i["riverName"] if "riverName" in i else ""
        long = i["long"]
        lat = i["lat"]

        station = (label, river_name, long, lat)
        stations.append(station)
        text += station_to_text(station) + "\n"

        for j in i["measures"]:
            parameter = j["parameter"]

            try:
                reading_response = requests.get(j['@id'])
                unit = reading_response.json()["items"][0]["unitName"]
            except:
                unit = "" # For example, PH has no unit.

            try:
                reading_response = requests.get(j['@id'] + suffix)
                value = reading_response.json()["items"][0]["value"]
            except:
                continue

            measure = (parameter, value, unit)
            measures.append(measure)
            text += measure_to_text(measure) + "\n"

    return text, stations, measures

def get_realtime_data(location):
    long, lat = get_long_lat_from_postcode(location)
    text, stations, measures = get_hydrology_info(long, lat)

    return stations, measures, text

def generate_map(location, stations):
    long, lat = get_long_lat_from_postcode(location)

    map = folium.Map(location=(lat, long), zoom_start=11)

    folium.Marker(
        location=[lat, long],
        tooltip="",
        popup=location,
        icon=folium.Icon(icon="glyphicon-pushpin", color="red")
    ).add_to(map)

    for label, river_name, long, lat in stations:
        folium.Marker(
            location=[lat, long],
            tooltip=river_name,
            popup=label,
            icon=folium.Icon(icon="glyphicon-tint", color="blue")
        ).add_to(map)

    x = uuid.uuid4()
    map.save(f"map_{x}.html")

    return gr.HTML(f"<embed src='/gradio_api/file=./map_{x}.html' style='width:470px;height:470px;'>", visible=True)

def step_to_title(step):
    x = {
        1: "Hydrological setting",
        2: "Water resources and infrastructures",
        3: "Risks",
        4: "Impact of the new development",
        5: "Climate change resilience and implications",
        6: "Summary",
    }

    return x[step]

def step_to_simplified_titles(step):
    x = {
        1: "Hydrology",
        2: "Infrastructures",
        3: "Risks",
        4: "Impact",
        5: "Climate",
        6: "Summary",
    }

    return x[step]

def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        prompt += "<|" + message["role"] + "|>" + "\n" + message["content"] + "\n"
    prompt += "<|assistant|>\n"

    return prompt

def verify(verification_agent, title, response):
    prompt = verification_agent.get_prompt().format(title=title, response=response)
    messages = [{"role": "user", "content": prompt}]
    prompt = messages_to_prompt(messages)

    completion = verification_agent.generate(prompt)

    time.sleep(0.5)

    return "yes" in completion.lower()

def generate_report(agents, step, location, context, report, realtime_stations, realtime_measures, realtime_data, progress=gr.Progress()):
    if step == 0:
        progress(0, desc=f"Fetching real-time data...")
        stations, measures, text = get_realtime_data(location)
        return report, stations, measures, text

    simplified_title = step_to_simplified_titles(step)
    title = step_to_title(step)

    num_steps = 6
    max_num_attempts = 3

    if step == num_steps:
        agent = agents["summarisation_agent"]
        verif_agent = None
    else:
        agent = agents["generation_agent"]
        verif_agent = agents["verification_agent"]

    for cur_attempt in range(max_num_attempts):
        progress(0, desc=f"Generating {simplified_title} ({step}/{num_steps})")

        agent_params = agent.get_params()

        if cur_attempt > 0:
            # Change default parameters so that model generates a different output.
            agent_params = agent_params.copy()
            agent_params["decoding_method"] = "sample"
            agent_params["temperature"] = 0.7
            agent_params["top_k"] = 50

        prompt = agent.get_prompt().format(step=step, title=title, location=location, context=context, realtime_data=realtime_data, report=report)
        messages = [{"role": "user", "content": prompt}]
        prompt = messages_to_prompt(messages)

        completion = agent.generate(prompt, agent_params)

        progress(1, desc=f"Verifying {simplified_title} ({step}/{num_steps})")

        if not verif_agent or verify(verif_agent, title, completion):
            break

    report = report + "\n\n" + completion

    filename = "report.md"
    with open(filename, "w") as f:
        f.write(report)

    return report, realtime_stations, realtime_measures, realtime_data

def enable_post_report_buttons():
    return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

def disable_pre_report_buttons():
    return gr.update(interactive=False), gr.update(interactive=False)

def reset():
    return gr.update(value="", interactive=True), \
        gr.update(value="", interactive=True), \
        gr.update(value="", visible=False), \
        gr.update(visible=True), \
        gr.update(visible=False), \
        gr.update(visible=False), \
        gr.update(visible=False), \
        gr.update(visible=False), \
        gr.update(visible=False), \
        gr.update(visible=False), \
        gr.update(visible=False), \
        gr.update(visible=False), \
        gr.update(visible=False), \
        gr.update(visible=False), \
        gr.update(visible=False), \
        gr.update(visible=False)

def chat(message, history):
    filename = "report.md"
    with open(filename, "r") as f:
        report = f.read()

    messages = [{"role": "user", "content": report}] + history + [{"role": "user", "content": message}]
    prompt = messages_to_prompt(messages)
    completion = agents["chat_agent"].generate(prompt)

    return completion

def get_examples():
    return [
        "What would you suggest I do to mitigate the potential impacts of my project on the local water network?",
        "What is the last temperature recorded at the nearby water body? Has it fluctuated over the past 2 years?",
        "Tell me more about the expected climate change in the area.",
        "When was the last flood?",
    ]

def enable_checkboxes():
    return gr.update(value=False, visible=True), \
        gr.update(value=False, visible=True), \
        gr.update(value=False, visible=True), \
        gr.update(value=False, visible=True), \
        gr.update(value=False, visible=True), \
        gr.update(value=False, visible=True), \
        gr.update(value=False, visible=True)

def update_status(realtime_stations, realtime_measures):
    return gr.Checkbox(label=f"Realtime data ({len(realtime_stations)} stations, {len(realtime_measures)} measures)", interactive=False, visible=True, value=True)

def authenticate(watsonx_ai_url, watsonx_ai_api_key, watsonx_ai_project_id):
    print("Authenticating to IBM watsonx.ai...")

    credentials = Credentials(url=watsonx_ai_url, api_key=watsonx_ai_api_key)
    client = APIClient(credentials)
    client.set.default_project(watsonx_ai_project_id)

    return credentials

def setup_agents(credentials, project_id, prompt_manager):
    templates = prompt_manager.list()

    known_agents = ["generation_agent", "verification_agent", "summarisation_agent", "chat_agent"]
    assert set(known_agents).issubset(set(list(templates["NAME"])))

    agents = {}
    for agent_name in list(templates["NAME"]):
        template_id = templates.loc[templates["NAME"] == agent_name]["ID"].values[0]
        template = prompt_manager.load_prompt(template_id)
        agents[agent_name] = Agent(credentials=credentials, project_id=project_id, model_id=template.model_id, prompt=template.input_text, params=template.model_params)

    return agents

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walter Reports")
    parser.add_argument("--watsonx_ai_url", type=str, help="watsonx.ai URL", required=True)
    parser.add_argument("--watsonx_ai_api_key", type=str, help="watsonx.ai API key", required=True)
    parser.add_argument("--watsonx_ai_project_id", type=str, help="watsonx.ai project id", required=True)
    args = parser.parse_args()

    theme = gr.themes.Default(
        primary_hue="blue",
        radius_size="none",
    ).set(
        embed_radius='radius_xxs',
        block_radius='radius_xxs',
        checkbox_border_radius='radius_xxs',
        input_radius='radius_xxs',
        button_large_radius='radius_xxs',
        button_small_radius='radius_xxs',
        button_secondary_background_fill='*secondary_200'
    )

    credentials = authenticate(
        watsonx_ai_url=args.watsonx_ai_url,
        watsonx_ai_api_key=args.watsonx_ai_api_key,
        watsonx_ai_project_id=args.watsonx_ai_project_id)

    prompt_manager = PromptTemplateManager(
        credentials=credentials,
        project_id=args.watsonx_ai_project_id)

    agents = setup_agents(credentials, args.watsonx_ai_project_id, prompt_manager)

    with gr.Blocks(theme=theme, title="Walter Reports") as demo:
        gr.HTML("<img src='/gradio_api/file=./assets/walters-logo.png' alt='logo' width='192' height='192'>")
        gr.Markdown("# Walter Reports: generate your water assessment report!")

        with gr.Row():
            location = gr.Textbox(label="Location")
            context = gr.Textbox(label="Context")

        with gr.Row():
            with gr.Column(scale=2):
                report = gr.Markdown(label="Report", container=True, visible=False, min_height=64)
            with gr.Column(scale=1):
                map = gr.HTML("", visible=False)

                checkboxes = [gr.Checkbox(label="Realtime data", interactive=False, visible=False)]
                for i in range(1, 7):
                    checkboxes.append(gr.Checkbox(label=step_to_simplified_titles(i), interactive=False, visible=False))

        generate_report_button = gr.Button(value="Generate report", variant="primary")

        with gr.Group(visible=False) as chat_group:
            chat = gr.ChatInterface(fn=chat, type="messages", examples=get_examples())

        with gr.Row():
            download_button = gr.DownloadButton(label="Download report", value="report.md", visible=False, variant="primary")
            reset_button = gr.Button(value="Reset", visible=False)
            discuss_button = gr.Button(value="Discuss your report with Walter AI", visible=False)

        all_objects = [location, context, report, generate_report_button, download_button, reset_button, discuss_button, chat_group, map] + checkboxes

        realtime_stations = gr.State([])
        realtime_measures = gr.State([])
        realtime_data = gr.State([])

        generate_report_button.click(fn=lambda: gr.update(visible=False), inputs=None, outputs=generate_report_button) \
            .then(generate_map, inputs=[location, realtime_stations], outputs=map) \
            .then(fn=lambda: gr.update(visible=True), inputs=None, outputs=report) \
            .then(enable_checkboxes, inputs=None, outputs=checkboxes) \
            .then(functools.partial(generate_report, agents, 0), inputs=[location, context, report, realtime_stations, realtime_measures, realtime_data], outputs=[report, realtime_stations, realtime_measures, realtime_data]) \
            .then(update_status, inputs=[realtime_stations, realtime_measures], outputs=checkboxes[0]) \
            .then(generate_map, inputs=[location, realtime_stations], outputs=map) \
            .then(functools.partial(generate_report, agents, 1), inputs=[location, context, report, realtime_stations, realtime_measures, realtime_data], outputs=[report, realtime_stations, realtime_measures, realtime_data]) \
            .then(fn=lambda: gr.update(value=True), inputs=None, outputs=checkboxes[1]) \
            .then(functools.partial(generate_report, agents, 2), inputs=[location, context, report, realtime_stations, realtime_measures, realtime_data], outputs=[report, realtime_stations, realtime_measures, realtime_data]) \
            .then(fn=lambda: gr.update(value=True), inputs=None, outputs=checkboxes[2]) \
            .then(functools.partial(generate_report, agents, 3), inputs=[location, context, report, realtime_stations, realtime_measures, realtime_data], outputs=[report, realtime_stations, realtime_measures, realtime_data]) \
            .then(fn=lambda: gr.update(value=True), inputs=None, outputs=checkboxes[3]) \
            .then(functools.partial(generate_report, agents, 4), inputs=[location, context, report, realtime_stations, realtime_measures, realtime_data], outputs=[report, realtime_stations, realtime_measures, realtime_data]) \
            .then(fn=lambda: gr.update(value=True), inputs=None, outputs=checkboxes[4]) \
            .then(functools.partial(generate_report, agents, 5), inputs=[location, context, report, realtime_stations, realtime_measures, realtime_data], outputs=[report, realtime_stations, realtime_measures, realtime_data]) \
            .then(fn=lambda: gr.update(value=True), inputs=None, outputs=checkboxes[5]) \
            .then(functools.partial(generate_report, agents, 6), inputs=[location, context, report, realtime_stations, realtime_measures, realtime_data], outputs=[report, realtime_stations, realtime_measures, realtime_data]) \
            .then(fn=lambda: gr.update(value=True), inputs=None, outputs=checkboxes[6]) \
            .then(disable_pre_report_buttons, inputs=None, outputs=[location, context]) \
            .then(enable_post_report_buttons, inputs=None, outputs=[download_button, reset_button, discuss_button])

        reset_button.click(reset, inputs=None, outputs=all_objects)
        discuss_button.click(fn=lambda: gr.update(visible=True), inputs=None, outputs=chat_group)

    demo.launch(allowed_paths=["."])
