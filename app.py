#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:42:14 2024

@author: Yannick
"""

#Load packages
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import matplotlib


#Load data
cleaned_trade_data = pd.read_csv("cleaned_trade_data.csv")

# Specify a suffix for the value
value_suffix = "tonne"

# Customize layout, font, and colors
fontsize = 14  # Set font size of labels
fontfamily = "Helvetica"  # Set font family of plot's text
bgcolor = "SeaShell"  # Set the plot's background color (use color name or hex code)
link_opacity = 0.3  # Set a value from 0 to 1: the lower, the more transparent the links
node_colors = px.colors.qualitative.G10  # Define a list of hex color codes for nodes


cols = ['departure_region', 'arrival_region']  # Define the columns to use
weight = "weight_tonnes"

available_years = cleaned_trade_data['year'].unique()
slider_marks = {str(year): '' for year in available_years}

app = Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        options=[
                {"label": "Tea", "value": "Tea"},
                {"label": "Silk", "value": "Silk"},
                {"label": "Cloves", "value": "Cloves"},
                {"label": "Nutmeg", "value": "Nutmeg"},
                {"label": "Pepper", "value": "Pepper"},
                {"label": "Cinnamon", "value": "Cinnamon"},
                {"label": "Sugar", "value": "Sugar"},
                {"label": "Coffee", "value": "Coffee"},
                {"label": "Opium", "value": "Opium"}
            ],
            value='Pepper',
            id='dropdown-selection'
    ),
    
    html.Br(),
    
    dcc.Graph(id='graph-content'),
    html.Br(),
    html.Br(),
    dcc.Slider(
        id='year-slider',
        min=min(available_years),
        max=max(available_years),
        step=1,
        value=min(available_years),  # Set initial value to the minimum year
        marks=slider_marks,
        tooltip={
        "always_visible": True,
        "style": {"color": "LightSteelBlue", "fontSize": "20px"},
    }
)
])

@app.callback(
    [Output('graph-content', 'figure'),
     Output('year-slider', 'marks')],
    [Input('dropdown-selection', 'value'),
     Input('year-slider', 'value')]
)

def update_graph(selected_product, selected_year):
    df_allyears = cleaned_trade_data[cleaned_trade_data["product_category"]==selected_product]
       
    df = df_allyears[df_allyears["year"] == selected_year]

    # Get unique years for the selected commodity
    available_years = df_allyears['year'].unique()

    # Generate marks for the slider based on available years
    slider_marks = {str(year): '' for year in available_years}
    
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title='No data')
        
        # If DataFrame is empty, return a warning message or placeholder figure
        
        return fig, slider_marks
    
    
    else:
        
        s = []  # This will hold the source nodes
        t = []  # This will hold the target nodes
        v = []  # This will hold the flow volumes between the source and target nodes
        labels = np.unique(df[cols].values)  # Collect all the node labels

        # Get all the links between two nodes in the data and their corresponding values
        for c in range(len(cols) - 1):
            s.extend(df[cols[c]].tolist())
            t.extend(df[cols[c + 1]].tolist())
            v.extend(df[weight].tolist())
        links = pd.DataFrame({"source": s, "target": t, "weight": v})  
        links = links.groupby(["source", "target"], as_index=False).agg({"weight": "sum"})

        # Convert list of colors to RGB format to override default gray link colors
        colors = [matplotlib.colors.to_rgb(i) for i in node_colors]  

        # Create objects to hold node/label and link colors
        label_colors, links["link_c"] = [], 0

        # Loop through all the labels to specify color and to use label indices
        c, max_colors = 0, len(colors)  # To loop through the colors array
        for l in range(len(labels)):
            label_colors.append(colors[c])
            link_color = colors[c] + (link_opacity,)  # Make link more transparent than the node
            links.loc[links.source == labels[l], ["link_c"]] = "rgba" + str(link_color)
            links = links.replace({labels[l]: l})  # Replace node labels with the label's index
            if c == max_colors - 1:
                c = 0
            else:
                c += 1

        # Convert colors into RGB string format for Plotly
        label_colors = ["rgb" + str(i) for i in label_colors]

        # Define a Plotly Sankey diagram
        fig = go.Figure( 
            data=[
                go.Sankey(
                    valuesuffix=value_suffix,
                    node=dict(label=labels, color=label_colors),
                    link=dict(
                        source=links["source"],
                        target=links["target"],
                        value=links["weight"],
                        color=links["link_c"],
                    ),
                )
            ]
        )

        # Customize plot based on earlier values
        fig.update_layout(
            title_text = f"Sankey diagram of trade flows in the VOC for {selected_product} in {selected_year}",
            font_size=fontsize,
            font_family=fontfamily,
            width=1200,
            height=600,
            paper_bgcolor=bgcolor,
            title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},  # Centers title
        )

        return fig, slider_marks

if __name__ == '__main__':
    app.run(jupyter_mode="external", port=8090)
