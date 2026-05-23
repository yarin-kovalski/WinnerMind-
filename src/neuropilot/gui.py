"""Optional Gradio launcher for WinnerMind training artifacts."""

from __future__ import annotations

from queue import Queue
from threading import Thread

from neuropilot.config import REPORTS_DIR
from neuropilot.demo import predict_random_shot
from neuropilot.genetic_algorithm import train, write_best_shot


def launch() -> None:
    import gradio as gr

    def progress_html(generation: int, total: int, best: float = 0.0, average: float = 0.0) -> str:
        percent = 0 if total == 0 else round((generation / total) * 100)
        return f"""
        <div class="wm-progress">
          <div class="wm-progress-top">
            <strong>Generation {generation} / {total}</strong>
            <span>{percent}%</span>
          </div>
          <div class="wm-progress-track">
            <div class="wm-progress-fill" style="width: {percent}%"></div>
          </div>
          <div class="wm-progress-stats">Best fitness {best:.2f} | Average fitness {average:.2f}</div>
        </div>
        """

    def train_action():
        events: Queue[tuple[str, int, int, float, float] | tuple[str, object]] = Queue()

        def on_generation(generation: int, total: int, best: float, average: float) -> None:
            events.put(("generation", generation, total, best, average))

        def worker() -> None:
            try:
                result = train(on_generation=on_generation)
                events.put(("done", result))
            except Exception as error:  # pragma: no cover - GUI safety
                events.put(("error", error))

        Thread(target=worker, daemon=True).start()
        yield (
            "Training started. The green bar updates after each generation.",
            None,
            None,
            progress_html(0, 1),
        )

        while True:
            event = events.get()
            if event[0] == "generation":
                _, generation, total, best, average = event
                yield (
                    f"Generation {generation}/{total}: best fitness {best:.2f}, average fitness {average:.2f}",
                    str(REPORTS_DIR / "fitness.png"),
                    str(REPORTS_DIR / "best_shot.json"),
                    progress_html(generation, total, best, average),
                )
            elif event[0] == "done":
                best = event[1]
                best_shot = best.best_result
                yield (
                    f"Training complete. Best average fitness: {best.fitness:.2f}. "
                    f"Best shot fitness: {best_shot.fitness:.2f}.",
                    str(REPORTS_DIR / "fitness.png"),
                    str(REPORTS_DIR / "best_shot.json"),
                    progress_html(1, 1, best.fitness, best_shot.fitness),
                )
                break
            else:
                error = event[1]
                yield (f"Training failed: {error}", None, None, progress_html(0, 1))
                break

    def random_shot_action() -> tuple[str, str]:
        try:
            result = predict_random_shot()
        except RuntimeError as error:
            return str(error), str(REPORTS_DIR / "best_shot.json")
        write_best_shot(result, REPORTS_DIR / "best_shot.json")
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
    .wm-progress {
        padding: 14px;
        border: 1px solid rgba(116, 245, 157, 0.45);
        border-radius: 8px;
        background: rgba(13, 28, 21, 0.78);
        color: #eafff0;
        box-shadow: 0 0 24px rgba(25, 195, 125, 0.14);
    }
    .wm-progress-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 9px;
        color: #b7ff3c;
    }
    .wm-progress-track {
        width: 100%;
        height: 18px;
        overflow: hidden;
        border-radius: 999px;
        background: #1b2420;
        border: 1px solid rgba(116, 245, 157, 0.28);
    }
    .wm-progress-fill {
        height: 100%;
        min-width: 8px;
        border-radius: 999px;
        background:
            repeating-linear-gradient(
                45deg,
                rgba(255,255,255,0.18) 0,
                rgba(255,255,255,0.18) 10px,
                rgba(255,255,255,0.02) 10px,
                rgba(255,255,255,0.02) 20px
            ),
            linear-gradient(90deg, #19c37d, #b7ff3c);
        background-size: 34px 34px, auto;
        box-shadow: 0 0 20px rgba(25, 195, 125, 0.5);
        transition: width 260ms ease;
        animation: wm-stripes 900ms linear infinite;
    }
    .wm-progress-stats {
        margin-top: 8px;
        color: #9eeeb8;
        font-size: 13px;
    }
    @keyframes wm-stripes {
        from { background-position: 0 0, 0 0; }
        to { background-position: 34px 0, 0 0; }
    }
    .gradio-container [class*="progress"],
    .gradio-container [class*="progress"] *,
    .gradio-container [role="progressbar"],
    .gradio-container [role="progressbar"] *,
    .progress-text,
    .wrap.svelte-1ipelgc {
        color: #74f59d !important;
    }
    .gradio-container [class*="progress-level"],
    .gradio-container [class*="progress-level"] *,
    .gradio-container [class*="progress-bar"],
    .gradio-container [class*="progress-bar"] *,
    .gradio-container [role="progressbar"],
    .progress-level,
    .progress-level-inner {
        background: linear-gradient(90deg, #19c37d, #b7ff3c) !important;
        background-color: #19c37d !important;
    }
    .gradio-container [class*="progress-level"],
    .gradio-container [class*="progress-bar"],
    .gradio-container [role="progressbar"],
    .progress-level {
        border-color: rgba(116, 245, 157, 0.55) !important;
        box-shadow: 0 0 20px rgba(25, 195, 125, 0.32) !important;
    }
    .gradio-container .wrap > div[style*="width"],
    .gradio-container .wrap > div[style*="transform"],
    .gradio-container .wrap > div[style*="translate"] {
        background: linear-gradient(90deg, #19c37d, #b7ff3c) !important;
        background-color: #19c37d !important;
    }
    """

    with gr.Blocks(title="WinnerMind Tennis Shot Strategy AI", css=css) as app:
        gr.Markdown("# WinnerMind Tennis Shot Strategy AI")
        gr.Markdown("Train a Keras model with a genetic algorithm, then open the 3D court visualization.")
        train_button = gr.Button("Train")
        train_text = gr.Textbox(label="Training status")
        train_progress = gr.HTML(value=progress_html(0, 1), label="Training progress")
        fitness_image = gr.Image(label="Fitness chart")
        best_json = gr.File(label="Best shot JSON")
        train_button.click(train_action, outputs=[train_text, fitness_image, best_json, train_progress])
        gr.Markdown("## Random incoming ball demo")
        random_button = gr.Button("Generate WinnerMind Return")
        random_text = gr.Textbox(label="Shot decision")
        random_json = gr.File(label="Updated 3D shot JSON")
        random_button.click(random_shot_action, outputs=[random_text, random_json])

    app.launch()


if __name__ == "__main__":
    launch()
