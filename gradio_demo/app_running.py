import os
import numpy as np
import random
import time
import gradio as gr
from gradio_demo.runner import Runner
import matplotlib.pyplot as plt

def show_mask(mask, ax, color='blue'):
    if color == 'blue':
        # reference, blue
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    else:
        # target, green
        color = np.array([78 / 255, 238 / 255, 148 / 255, 0.6])

    # if random_color:
    #     color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    # else:
    #     color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_img_point_box_mask(img, input_point=None, input_label=None, box=None, masks=None, save_path=None, mode='mask', color='blue'):

    if mode == 'point':
        # point
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        show_points(input_point, input_label, plt.gca())
        plt.axis('on')
        plt.savefig(save_path, bbox_inches='tight')
    elif mode == 'box':
        # box
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        show_box(box, plt.gca())
        plt.axis('on')
        plt.savefig(save_path, bbox_inches='tight')
    else:
        # mask
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        show_mask(masks, plt.gca(), color=color)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def create_oss_demo(
        runner: Runner,
        pipe: None = None
) -> gr.Blocks:

    examples = [
        ['./gradio_demo/images/horse1.png', './gradio_demo/images/horse2.png', './gradio_demo/images/horse3.png'],
        ['./gradio_demo/images/hmbb1.png', './gradio_demo/images/hmbb2.png', './gradio_demo/images/hmbb3.png'],
        ['./gradio_demo/images/earth1.png', './gradio_demo/images/earth2.png', './gradio_demo/images/earth3.png'],
        ['./gradio_demo/images/elephant1.png', './gradio_demo/images/elephant2.png', './gradio_demo/images/elephant3.png'],
        ['./gradio_demo/images/dinosaur1.png', './gradio_demo/images/dinosaur2.png', './gradio_demo/images/dinosaur3.png'],
    ]

    with gr.Blocks() as oss_demo:
        with gr.Column():

            # inputs
            with gr.Row():
                img_input_prompt = gr.ImageMask(label='Prompt (提示图)')
                img_input_target1 = gr.Image(label='Target 1 (测试图1)')
                img_input_target2 = gr.Image(label='Target 2 (测试图2)')

            version = gr.inputs.Radio(['version 1 (🔺 multiple instances  🔻 whole, 🔻 part)',
                                       'version 2 (🔻 multiple instances  🔺 whole, 🔻 part)',
                                       'version 3 (🔻 multiple instances  🔻 whole, 🔺 part)'],
                                      type="value", default='version 1 (🔺 whole, 🔻 part)',
                                      label='Multiple Instances (version 1), Single Instance (version 2), Part of a object (version 3)')

            with gr.Row():
                submit1 = gr.Button("提交 (Submit)")
                clear = gr.Button("清除 (Clear)")
            info = gr.Text(label="Processing result: ", interactive=False)

            # decision
            K = gr.Slider(0, 10, 10, step=1, label="Controllable mask output", interactive=True)
            submit2 = gr.Button("提交 (Submit)")

            # outputs
            with gr.Row():
                img_output_pmt = gr.Image(label='Prompt (提示图)')
                img_output_tar1 = gr.Image(label='Output 1 (输出图1)')
                img_output_tar2 = gr.Image(label='Output 2 (输出图2)')

        # images
        gr.Examples(
            examples=examples,
            fn=runner.inference_oss_ops,
            inputs=[img_input_prompt, img_input_target1, img_input_target2],
            outputs=info
        )

        submit1.click(
            fn=runner.inference_oss_ops,
            inputs=[img_input_prompt, img_input_target1, img_input_target2, version],
            outputs=info
        )
        submit2.click(
            fn=runner.controllable_mask_output,
            inputs=K,
            outputs=[img_output_pmt, img_output_tar1, img_output_tar2]
        )

        clear.click(
            fn=runner.clear_fn,
            inputs=None,
            outputs=[img_input_prompt, img_input_target1, img_input_target2, info, img_output_pmt, img_output_tar1, img_output_tar2],
            queue=False
        )

    return oss_demo


def create_vos_demo(
        runner: Runner,
        pipe: None = None
) -> gr.Interface:

    raise NotImplementedError

def create_demo(
        runner: Runner,
        pipe: None = None
) -> gr.TabbedInterface:

    title = "Matcher🎯: Segment Anything with One Shot Using All-Purpose Feature Matching<br> \
    <div align='center'> \
    <h2><a href='https://arxiv.org/abs/2305.13310' target='_blank' rel='noopener'>[paper]</a> \
    <a href='https://github.com/aim-uofa/Matcher' target='_blank' rel='noopener'>[code]</a></h2> \
    <h2>Matcher can segment anything with one shot by integrating an all-purpose feature extraction model and a class-agnostic segmentation model.</h2> \
    <br> \
    </div> \
    "

    oss_demo = create_oss_demo(runner=runner, pipe=pipe)
    # vos_demo = create_vos_demo(runner=runner, pipe=pipe)
    demo = gr.TabbedInterface(
        [oss_demo,],
        ['OSS+OPS',], title=title)
    return demo


if __name__ == '__main__':
    pipe = None
    HF_TOKEN = os.getenv('HF_TOKEN')
    runner = Runner(HF_TOKEN)
    # runner = None
    demo = create_demo(runner, pipe)
    demo.launch(enable_queue=False)