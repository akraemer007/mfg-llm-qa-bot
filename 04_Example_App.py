# Databricks notebook source
# MAGIC %md You may find this notebook on https://github.com/databricks-industry-solutions/mfg-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ##Example Application
# MAGIC
# MAGIC This is an example application that you can leverage to make an api call to the model that's now hosted in Databricks model serving. This application can be hosted locally, on Huggingface Spaces, on a Databricks VM or any other VM that can run Python. For more info on gradio, visit https://www.gradio.app/guides/quickstart
# MAGIC
# MAGIC
# MAGIC <p>
# MAGIC     <img src="https://github.com/databricks-industry-solutions/mfg-llm-qa-bot/raw/main/images/Example-App.png" width="700" />
# MAGIC </p>
# MAGIC

# COMMAND ----------

# MAGIC %pip install gradio

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

#TODO figure out what I need from configs

# COMMAND ----------

# MAGIC %run ./_resources/00-init $catalog=akraemer $db=custom_llm_demo $reset_all_data=false

# COMMAND ----------

# TODO pass this in
company_name_full = "American Airlines"
company_name = company_name_full.replace(' ', '_').lower()

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json
import gradio as gr

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
databricks_url = ctx.apiUrl().getOrElse(None)
endpoint=f"{company_name}_financial_transcript"
endpoint_url = f"""{databricks_url}/serving-endpoints/{endpoint}/invocations"""

serving_client = EndpointApiClient()
serving_client.get_inference_endpoint(endpoint)

def create_tf_serving_json(data):
    return {
        "inputs": {name: data[name].tolist() for name in data.keys()}
        if isinstance(data, dict)
        else data.tolist()
    }


def score_model(question):

    response = requests.post(
        endpoint_url,
        json={"dataframe_split": {"data": [question]}},
        headers=serving_client.headers,
    ).json()
    return response["predictions"]


def greet(question):
    # TODO add source stuff here
    data = score_model(question)

    answer = data[0]["answer"].replace("\n", " ").replace("<br/>", " ").replace("<br/><br/>", " ").replace("Answer: ", "")

    return answer

def srcshowfn(chkbox):

    vis = True if chkbox == True else False
    print(vis)
    return gr.Textbox.update(visible=vis)


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        gr.HTML(
            show_label=False,
            value="<img src='https://databricks.gallerycdn.vsassets.io/extensions/databricks/databricks/0.3.15/1686753455931/Microsoft.VisualStudio.Services.Icons.Default' height='30' width='30'/><div font size='1'>Manufacturing</div>",
        )
    with gr.Row():
        gr.Markdown(
            f"""
            # {company_name_full} Financials Q&A Bot
            This bot has been trained on publicly availible data from {company_name_full}. For the purposes of this demo, we have only downloaded the chemicals that start with the most recent quarter's transcript. The application simply makes an API call to the model that's hosted in Databricks.
            """
        )
    with gr.Row():
        input_question = gr.Textbox(
            placeholder="ex. What was the revenue in Q3? or What headwinds is the company facing?",
            label="Question",
        )
    with gr.Row():
        output = gr.Textbox(label="Prediction")
        greet_btn = gr.Button("Respond", size="sm", scale=0)#.style(height=20)
    # with gr.Row(): # TODO add sources
    #     srcshow = gr.Checkbox(value=False, label='Show sources')
    # with gr.Row():
    #     outputsrc = gr.Textbox(label="Sources", visible=False)

    # srcshow.change(srcshowfn, inputs=srcshow, outputs=outputsrc)
    greet_btn.click(fn=greet, inputs=[input_question], outputs=[output], api_name="greet")

demo.launch(share=True)

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC |  Gradio | Build Machine Learning Web Apps in Python |  Apache Software License  |   https://pypi.org/project/gradio/ |

# COMMAND ----------

greet_btn.
