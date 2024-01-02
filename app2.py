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
caption_model_name = 'blip-large' #@param ["blip-base", "blip-large", "git-large-coco"]
clip_model_name = 'ViT-L-14/openai' #@param ["ViT-L-14/openai", "ViT-H-14/laion2b_s32b_b79k"]
import gradio as gr
from clip_interrogator import Config, Interrogator
config = Config()
config.clip_model_name = clip_model_name
config.caption_model_name = caption_model_name
ci = Interrogator(config)
def image_analysis(image):
    image = image.convert('RGB')
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
    ci.config.chunk_size = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    ci.config.flavor_intermediate_count = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    image = image.convert('RGB')
    if mode == 'best':
        return ci.interrogate(image)
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    elif mode == 'fast':
        return ci.interrogate_fast(image)
    elif mode == 'negative':
        return ci.interrogate_negative(image)
#@title Image to prompt! 🖼️ -> 📝
title = "Prompt - Воровайка(CLIP)"
def prompt_tab():
    with gr.Column():
        with gr.Row():
            image = gr.inputs.Image(label="Image")
            with gr.Column():
                mode = gr.inputs.Radio(['best', 'fast', 'classic', 'negative'], label='Mode', default='best')
            button = gr.Button("Generate prompt")
        prompt = gr.outputs.Textbox(label="Prompt")
    return gr.Interface(inputs=[image, mode], outputs=prompt, titled_button=True, title=title, layout='vertical')
def analyze_tab():
    with gr.Column():
        with gr.Row():
            image = gr.inputs.Image(label="Image")
        with gr.Row():
            medium = gr.outputs.Label(label="Medium", type="auto", max_length=5)
            artist = gr.outputs.Label(label="Artist", type="auto", max_length=5)
            movement = gr.outputs.Label(label="Movement", type="auto", max_length=5)
            trending = gr.outputs.Label(label="Trending", type="auto", max_length=5)
            flavor = gr.outputs.Label(label="Flavor", type="auto", max_length=5)
    return gr.Interface(inputs=image, outputs=[medium, artist, movement, trending, flavor], title="Analyze", layout='vertical')
ui = gr.Interface(
    fn=None,
    title=title,
    layout="vertical",
    description="Изображение в текстовое описание или анализ",
    inputs=[gr.Interface(fn=prompt_tab, title="Prompt", layout="vertical")],
    outputs=[gr.Interface(fn=analyze_tab, title="Analyze", layout="vertical")]
)
ui.launch(inbrowser=True)
