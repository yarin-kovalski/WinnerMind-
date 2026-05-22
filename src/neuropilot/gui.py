"""Optional Gradio launcher for AceMind training artifacts."""

from __future__ import annotations

from neuropilot.config import REPORTS_DIR
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

    with gr.Blocks(title="AceMind Tennis Shot Strategy AI") as app:
        gr.Markdown("# AceMind Tennis Shot Strategy AI")
        gr.Markdown("Train a Keras model with a genetic algorithm, then open the 3D court visualization.")
        train_button = gr.Button("Train")
        train_text = gr.Textbox(label="Training status")
        fitness_image = gr.Image(label="Fitness chart")
        best_json = gr.File(label="Best shot JSON")
        train_button.click(train_action, outputs=[train_text, fitness_image, best_json])

    app.launch()


if __name__ == "__main__":
    launch()
