import os
import subprocess
def setup():
    install_cmds = [
        ['pip', 'install', 'gradio'],
        ['pip', 'install', 'open_clip_torch'],
        ['pip', 'install', 'clip-interrogator'],
    ]
    for cmd in install_cmds:
        print(subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode('utf-8'))
setup()
import gradio as gr
from clip_interrogator import Config, Interrogator
config = Config()
config.clip_model_name = 'ViT-L-14/openai'
config.caption_model_name = 'blip-large'
ci = Interrogator(config)
def analyze_image(image):
    image = image.convert("RGB")
    medium_ranks, artist_ranks, movement_ranks, trending_ranks, flavor_ranks = image_analysis(image)
    return {
        "Medium": medium_ranks,
        "Artist": artist_ranks,
        "Movement": movement_ranks,
        "Trending": trending_ranks,
        "Flavor": flavor_ranks
    }
def image_analysis(image):
    image_features = ci.image_to_features(image)
    top_mediums = ci.mediums.rank(image_features, 5)
    top_artists = ci.artists.rank(image_features, 5)
    top_movements = ci.movements.rank(image_features, 5)
    top_trendings = ci.trendings.rank(image_features, 5)
    top_flavors = ci.flavors.rank(image_features, 5)
    medium_ranks = {medium: sim for medium, sim in zip(top_mediums, ci.similarities(image_features, top_mediums))}
    artist_ranks = {artist: sim for artist, sim in zip(top_artists, ci.similarities(image_features, top_artists))}
    movement_ranks = {movement: sim for movement, sim in zip(top_movements, ci.similarities(image_features, top_movements))}
    trending_ranks = {trending: sim for trending, sim in zip(top_trendings, ci.similarities(image_features, top_trendings))}
    flavor_ranks = {flavor: sim for flavor, sim in zip(top_flavors, ci.similarities(image_features, top_flavors))}
    return medium_ranks, artist_ranks, movement_ranks, trending_ranks, flavor_ranks
def image_to_prompt(image, mode):
    image = image.convert("RGB")
    if mode == 'best':
        return ci.interrogate(image)
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    elif mode == 'fast':
        return ci.interrogate_fast(image)
    elif mode == 'negative':
        return ci.interrogate_negative(image)
def prompt_tab(image, mode):
    prompt = image_to_prompt(image, mode)
    return prompt
def analyze_tab(image):
    results = analyze_image(image)
    return results
input_image = gr.inputs.Image(label="Image")
input_mode = gr.inputs.Radio(["best", "fast", "classic", "negative"], label="Mode", default="best")
output_prompt = gr.outputs.Textbox(label="Prompt")
output_analysis_medium = gr.outputs.Label(label="Medium", type="auto")
output_analysis_artist = gr.outputs.Label(label="Artist", type="auto")
output_analysis_movement = gr.outputs.Label(label="Movement", type="auto")
output_analysis_trending = gr.outputs.Label(label="Trending", type="auto")
output_analysis_flavor = gr.outputs.Label(label="Flavor", type="auto")
prompt_interface = gr.Interface(prompt_tab, inputs=[input_image, input_mode], outputs=output_prompt, title="Prompt")
analyze_interface = gr.Interface(analyze_tab, inputs=input_image, outputs=[output_analysis_medium, output_analysis_artist, output_analysis_movement, output_analysis_trending, output_analysis_flavor], title="Analyze")
interface = gr.Interface(
    fn=None,
    title="Мой интерфейс",
    layout="vertical",
    description="Изображение в текстовое описание или анализ",
    inputs=[gr.Interface(fn=prompt_tab, title="Prompt", layout="vertical")],
    outputs=[gr.Interface(fn=analyze_tab, title="Analyze", layout="vertical")]
)
interface.launch(inbrowser=True)
