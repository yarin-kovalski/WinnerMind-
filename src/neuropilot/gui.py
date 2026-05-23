"""Optional Gradio launcher for WinnerMind training artifacts."""

from __future__ import annotations

from neuropilot.config import REPORTS_DIR
from neuropilot.demo import predict_random_shot
from neuropilot.genetic_algorithm import train


def launch() -> None:
    import gradio as gr

    def train_action() -> tuple[str, str, str]:
        best = train()
        best_shot = best.best_result
        return (
            f"Training complete. Best average fitness: {best.fitness:.2f}. "
            f"Best shot fitness: {best_shot.fitness:.2f}.",
            str(REPORTS_DIR / "fitness.png"),
            str(REPORTS_DIR / "best_shot.json"),
        )

    def random_shot_action() -> tuple[str, str]:
        try:
            result = predict_random_shot()
        except RuntimeError as error:
            return str(error), str(REPORTS_DIR / "best_shot.json")
        data = result.to_visualization_dict()
        summary = (
            f"Incoming ball: x={data['incomingX']}m, z={data['incomingZ']}m, "
            f"height={data['incomingHeight']}m, speed={data['incomingSpeed']}m/s, spin={data['incomingSpin']}.\n"
            f"Opponent: x={data['opponentX']}m, z={data['opponentZ']}m, "
            f"running vx={data['opponentVx']}, vz={data['opponentVz']}.\n"
            f"WinnerMind return: power={data['power']}, launch={data['launchAngleDeg']} deg, "
            f"arc={data['arcHeight']}m, spin={data['spin']}, target=({data['targetX']}, {data['targetZ']}).\n"
            f"Fitness={data['fitness']}, clears net={data['clearedNet']}, in court={data['inCourt']}."
        )
        return summary, str(REPORTS_DIR / "best_shot.json")

    css = """
    .progress-text, .wrap.svelte-1ipelgc {
        color: #74f59d !important;
    }
    .progress-level-inner {
        background: linear-gradient(90deg, #19c37d, #b7ff3c) !important;
    }
    .progress-level {
        border-color: rgba(116, 245, 157, 0.55) !important;
    }
    """

    with gr.Blocks(title="WinnerMind Tennis Shot Strategy AI", css=css) as app:
        gr.Markdown("# WinnerMind Tennis Shot Strategy AI")
        gr.Markdown("Train a Keras model with a genetic algorithm, then open the 3D court visualization.")
        train_button = gr.Button("Train")
        train_text = gr.Textbox(label="Training status")
        fitness_image = gr.Image(label="Fitness chart")
        best_json = gr.File(label="Best shot JSON")
        train_button.click(train_action, outputs=[train_text, fitness_image, best_json])
        gr.Markdown("## Random incoming ball demo")
        random_button = gr.Button("Generate WinnerMind Return")
        random_text = gr.Textbox(label="Shot decision")
        random_json = gr.File(label="Updated 3D shot JSON")
        random_button.click(random_shot_action, outputs=[random_text, random_json])

    app.launch()


if __name__ == "__main__":
    launch()
