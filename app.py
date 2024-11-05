import os
import webbrowser

import matplotlib.pyplot as plt
from h2o_wave import app, data, main, Q, ui
from wordcloud import WordCloud
from pyspark.ml.tuning import CrossValidatorModel
from my_pipeline.lr_pipeline import LRPipeline

pipeline: LRPipeline
model: CrossValidatorModel 


def generate_wordcloud(text):
    wordcloud = WordCloud(width=2600, height=900, background_color='white').generate(text)

    plt.figure(figsize=(18, 6), facecolor=None)
    plt.imshow(wordcloud)

    plt.title('Word Cloud Created Based on Provided Lyrics', fontsize=14)

    temp_img_path = 'wordcloud.png'
    plt.savefig(temp_img_path)

    plt.close()

    return temp_img_path


def on_startup():
    global pipeline
    global model

    pipeline = LRPipeline()
    model = CrossValidatorModel.load("./model/")
    webbrowser.open("http://localhost:10101/")


def on_shutdown():
    pipeline.stop_pipeline()


@app("/", on_startup=on_startup, on_shutdown=on_shutdown)
async def serve(q: Q) -> None:
    if not q.client.initialized:
        q.client.initialized = True
        q.page["meta"] = ui.meta_card(
            "",
            title="MLlib",
            theme="h2o-dark",
            layouts=[
                ui.layout(
                    breakpoint="xs",
                    zones=[
                        ui.zone(
                            name="body",
                            direction=ui.ZoneDirection.ROW,
                            zones=[
                                ui.zone(name="left"),
                                ui.zone(
                                    name="right",
                                    zones=[
                                        ui.zone(
                                            "content",
                                            direction=ui.ZoneDirection.ROW,
                                            wrap=ui.ZoneWrap.STRETCH,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                )
            ],
        )

    q.page["left"] = ui.form_card(
        box=ui.box(zone="left", width="500px", height="100%"),
        items=[
            ui.textbox(
                name="input_lyrics",
                value=q.args.input_lyrics,
                label="Enter your lyrics",
                multiline=True,
                height="700px",
            ),
            ui.frame(" ", height="120px"),
            ui.buttons(
                [ui.button(name="predict", label="Predict", primary=True)],
                justify="end",
            ),
        ]
    )

    if q.args.predict and q.args.input_lyrics:
        if q.page["result"]:
            del q.page["wordcloud"]
            del q.page["result"]
            del q.page["bar_chart"]

        wordcloud_image_path = generate_wordcloud(q.args.input_lyrics)
        path = (await q.site.upload([wordcloud_image_path]))[0]
        os.remove(wordcloud_image_path)
        q.page['wordcloud'] = ui.image_card(box=ui.box(zone="content", width="1200px"), title='', path=path)

        q.page["loading"] = ui.form_card(
            box=ui.box(zone="content", width="1200px"),
            title="",
            items=[
                ui.progress(
                    label="Predicting...",
                ),
            ],
        )

        await q.page.save()

        prediction, probabilities = pipeline.predict_for_unknown_lyrics(
            q.args.input_lyrics, model
        )

        del q.page["loading"]

        await q.page.save()

        if max(probabilities.values()) < 0.3:
            prediction = f"UNKNOWN since {max(probabilities.values())} < 0.3"
        else:
            prediction = prediction.upper()

        q.page["result"] = ui.form_card(
            box=ui.box(zone="content", width="1200px"),
            items=[
                ui.text_xl(f"RESULT: <span style='color:red'>{prediction}</span>"),
            ]
        )

        q.page["bar_chart"] = ui.plot_card(
            box=ui.box(zone="content", width="1200px", height="440px"),
            title="The distribution of probabilities associated with the genres for given lyrics",
            data=data(
                'genre probability',
                len(probabilities),
                rows=[
                    (genre.capitalize(), probability * 100)
                    for genre, probability in probabilities.items()
                ],
            ),
            plot=ui.plot(
                marks=[
                    ui.mark(
                        type='interval',
                        x='=genre',
                        y='=probability',
                        y_min=0,
                        fill_color="#36BFFA",
                        stroke_color="#5E5E5E",
                    )
                ]
            ),
        )

    await q.page.save()
