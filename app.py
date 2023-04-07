
# @title ⬇️🖼️
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import gradio as gr
import torch
import random
import translators as ts

state = None
current_steps = 25
attn_slicing_enabled = True
mem_eff_attn_enabled = False
device = "cuda" if torch.cuda.is_available() else "cpu"
device_dict = {"cuda": 0, "cpu": -1}

model_id = 'prompthero/openjourney-v4'

scheduler = DPMSolverMultistepScheduler.from_pretrained(
    model_id, subfolder="scheduler")

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    # revision="fp16" if torch.cuda.is_available() else "fp32",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    scheduler=scheduler
)
pipe.enable_attention_slicing()
pipe = pipe.to(device)
if mem_eff_attn_enabled:
    pipe.enable_xformers_memory_efficient_attention()

modes = {
    'txt2img': 'Text to Image'
}
current_mode = modes['txt2img']


def error_str(error, title="Error"):
    return f"""#### {title}
            {error}""" if error else ""


def update_state(new_state):
    global state
    state = new_state


def update_state_info(old_state):
    visible = False
    if state:
        visible = True
    if state and state != old_state:
        return gr.update(value=state, visible=visible)


def set_mem_optimizations(pipe):
    if attn_slicing_enabled:
        pipe.enable_attention_slicing()
    else:
        pipe.disable_attention_slicing()

    if mem_eff_attn_enabled:
        pipe.enable_xformers_memory_efficient_attention()
    # else:
    #   pipe.disable_xformers_memory_efficient_attention()


def switch_attention_slicing(attn_slicing):
    global attn_slicing_enabled
    attn_slicing_enabled = attn_slicing


def switch_mem_eff_attn(mem_eff_attn):
    global mem_eff_attn_enabled
    mem_eff_attn_enabled = mem_eff_attn


def pipe_callback(step: int, timestep: int, latents: torch.FloatTensor):
    # \nTime left, sec: {timestep/100:.0f}")
    update_state(f"Hoàn thành {step}/{current_steps} bước tạo hình")


def inference(inf_mode, prompt, n_images, guidance, steps, isTrendy, isPopular, isNail, width=768, height=768, seed=0, neg_prompt=""):

    update_state("Đang chuẩn bị cho các bước tạo hình...")

    global current_mode
    if inf_mode != current_mode:
        pipe.to(device)
        current_mode = inf_mode

    if seed == 0:
        seed = random.randint(0, 2147483647)

    generator = torch.Generator(device).manual_seed(seed)
    prompt = prompt

    try:
        if inf_mode == modes['txt2img']:
            return txt_to_img(prompt, n_images, neg_prompt, guidance, steps, isTrendy, isPopular, isNail, width, height, generator, seed), gr.update(visible=False, value=None)
    except Exception as e:
        return None, gr.update(visible=True, value=error_str(e))


def txt_to_img(prompt, n_images, neg_prompt, guidance, steps, isTrendy, isPopular, isNail, width, height, generator, seed):
    text = ts.translate_text(prompt, from_language="vi")
    if isNail:
        text += ", nail art, close up"
    if isTrendy:
        text += ", trendy design"
    if isPopular:
        text += ", popular design"
    result = pipe(
        text,
        num_images_per_prompt=n_images,
        negative_prompt=neg_prompt,
        num_inference_steps=int(steps),
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator,
        callback=pipe_callback).images

    update_state(False)

    return result


def on_steps_change(steps):
    global current_steps
    current_steps = steps


with gr.Blocks(css="style.css", title="AI Nook") as demo:
    gr.HTML(
        f"""
          <div class="main-div" style="text-align: center;">
            <div>
              <h1>Nook AI Generation Art</h1>
            </div><br>
            Running on <b>{"GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"}</b>
          </div>
        """
    )
    with gr.Row():
        with gr.Column(scale=70):
            with gr.Group():
                with gr.Row():
                    with gr.Column(scale=1, variant="compact"):
                        prompt = gr.Textbox(label="Prompt", show_label=False, max_lines=2,
                                            placeholder=f"Nhập mô tả hình ảnh bạn muốn").style(container=False)
                        generate = gr.Button(value="Bắt đầu tạo hình").style(
                            rounded=(False, True, True, False))
                        with gr.Row():
                          isTrendy = gr.Checkbox(label="Trendy", value=True)
                          isPopular = gr.Checkbox(label="Phổ biến", value=False)
                          isNail = gr.Checkbox(label="Hình ảnh về Nails", value=False)
                        state_info = gr.Textbox(
                            label="State", show_label=False, max_lines=2, interactive=False, visible=False).style(container=False)
                        error_output = gr.Markdown(visible=False)
                    with gr.Column(scale=2):
                        gallery = gr.Gallery(label="Generated images", show_label=False).style(
                            grid=[1], height="auto", width="auto")

        with gr.Column(scale=30, visible=False):
            inf_mode = gr.Radio(label="Inference Mode", choices=list(
                modes.values()), value=modes['txt2img'])

            with gr.Group(visible=False):
                n_images = gr.Slider(
                    label="Number of images", value=1, minimum=1, maximum=4, step=1, visible=False)
                with gr.Row():
                    guidance = gr.Slider(
                        label="Guidance scale", value=7.5, maximum=15, visible=False)
                    steps = gr.Slider(
                        label="Steps", value=current_steps, minimum=2, maximum=100, step=1, visible=False)

                with gr.Row():
                    width = gr.Slider(label="Width", value=768,
                                      minimum=64, maximum=1024, step=8, visible=False)
                    height = gr.Slider(
                        label="Height", value=768, minimum=64, maximum=1024, step=8, visible=False)

                seed = gr.Slider(
                    0, 2147483647, label='Seed (0 = random)', value=0, step=1, visible=False)
                with gr.Accordion("Memory optimization", visible=False):
                    attn_slicing = gr.Checkbox(
                        label="Attention slicing (a bit slower, but uses less memory)", value=attn_slicing_enabled, visible=False)
                    # mem_eff_attn = gr.Checkbox(label="Memory efficient attention (xformers)", value=mem_eff_attn_enabled)

    steps.change(on_steps_change, inputs=[steps], outputs=[], queue=False)
    attn_slicing.change(lambda x: switch_attention_slicing(
        x), inputs=[attn_slicing], queue=False)
    # mem_eff_attn.change(lambda x: switch_mem_eff_attn(x), inputs=[mem_eff_attn], queue=False)

    inputs = [inf_mode, prompt, n_images, guidance, steps, isTrendy, isPopular, isNail, width, height, seed]
    outputs = [gallery, error_output]
    prompt.submit(inference, inputs=inputs, outputs=outputs)
    generate.click(inference, inputs=inputs, outputs=outputs)

    demo.load(update_state_info, inputs=state_info,
              outputs=state_info, every=0.5, show_progress=False)

    gr.HTML("""
    <div style="border-top: 1px solid #303030;">
      <br>
      <p style="display: flex; align-items: center;">Bạn có muốn áp dụng những bộ nails tuyệt vời trên? 
        <a href="https://nails.workwithnook.com">&nbsp; Đặt lịch ngay</a>
        <a href="https://nails.workwithnook.com" style="margin-left: 10px;" target="_blank"><img src="https://nails.workwithnook.com/assets/images/logo.svg" alt="Đặt lịch với Nook" style="height: 45px !important;width: 162px !important; display: unset;" ></a>
      </p>
      <br>
      <br>
    </div>
    """)

demo.queue(concurrency_count=5)
demo.launch(debug=False, share=False, height=768, favicon_path="favicon.svg")
