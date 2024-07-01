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

# Specify the value suffix (unit)
value_suffix = "tonne"

# Customize layout, font, and colors
fontsize = 14  # Set font size of labels
fontfamily = "Helvetica"  # Set font family of plot's text
bgcolor = "white"  # Set the plot's background color (use color name or hex code)
link_opacity = 0.3  # Set a value from 0 to 1: the lower, the more transparent the links
node_colors = px.colors.qualitative.G10  # Define a list of hex color codes for nodes

#Define the data to use
cols = ['departure_region', 'arrival_region']  # Define the columns to use for the values of the nodes
weight = "weight_tonnes" #Define the column for the weight

# Create a unique list of all nodes across the entire dataset
all_nodes = np.unique(cleaned_trade_data[cols].values)

# Map each unique node to a specific color
color_mapping = {node: color for node, color in zip(all_nodes, node_colors * (len(all_nodes) // len(node_colors) + 1))}

# Manually update the color of specific nodes
color_mapping['Batavia'] = 'red'  # Change Batavia to blue for better visibility

available_years = cleaned_trade_data['year'].unique()
slider_marks = {str(year): '' for year in available_years}

#Initiate dash app
app = Dash(__name__)

#Define the server for render
server = app.server

#Define layoyt of the app
app.layout = html.Div([
    #Add title and a description
    html.Div([
        html.H1("VOC Trade Flow Analysis", style={'fontFamily': 'Helvetica', 'fontSize': '32px'}),
        html.P([
            "This dashboard provides a visual representation of the trade flows in the VOC (Dutch East India Company) using a Sankey diagram. Explore the trade patterns by selecting a product and year. Hover over the flows to see the quantity of the flow. The tool was developed for the master's thesis of Yannick Egberink (2024) and is based on data from the ",
            html.A("Huygens Institute", href="https://resources.huygens.knaw.nl/das", target="_blank", style={'fontFamily': 'Helvetica', 'fontSize': '18px'}),
            "."
        ], style={'fontFamily': 'Helvetica', 'fontSize': '18px'})
    ], style={'textAlign': 'center', 'marginBottom': '20px', 'marginLeft': '10%', 'marginRight': '10%'}),

    #Add a dropdown menu
    html.Div(
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
            id='dropdown-selection',
            style={'width': '100%', 'fontFamily': 'Helvetica'}
        ),
        style={'display': 'flex', 'justifyContent': 'center', 'marginLeft': '5%', 'marginRight': '5%', 'width': '90%'}
    ),

    #Add the graph
    dcc.Graph(id='graph-content'),

    # add sankey diagram
    html.Div(
        dcc.Slider(
            id='year-slider',
            min=min(available_years),
            max=max(available_years),
            step=1,
            value=min(available_years),  # Set initial value to the minimum year
            marks=slider_marks,
            tooltip={
                "always_visible": True,
                "style": {"color": "LightSteelBlue", "fontSize": "20px", 'fontFamily': 'Helvetica'}
            }
        ),
        style={'marginLeft': '5%', 'marginRight': '5%'}
    )

])

#Define callback function
@app.callback(
    [Output('graph-content', 'figure'),
     Output('year-slider', 'marks')],
    [Input('dropdown-selection', 'value'),
     Input('year-slider', 'value')]
)

#Define function for the callback
def update_graph(selected_product, selected_year):

    #Filter the dataframe on the selected product
    df_allyears = cleaned_trade_data[cleaned_trade_data["product_category"]==selected_product]

    #Filter the dataframe on the selected year
    df = df_allyears[df_allyears["year"] == selected_year]

    # Get unique years for the selected commodity
    available_years = df_allyears['year'].unique()

    # Generate marks for the slider based on available years
    slider_marks = {str(year): '' for year in available_years}

    #Add if statement to return an empty placeholder with the prompt to select another year
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title='No data for this year, please select another year using the slider below.')
        
        # If DataFrame is empty, return a warning message or placeholder figure
        return fig, slider_marks

    #Else create the sankey diagram
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

        # Apply the fixed color mapping with opacity
        links["link_c"] = links["source"].map(lambda x: matplotlib.colors.to_rgba(color_mapping[x], link_opacity))
        
        # Convert colors into RGB string format for Plotly
        label_colors = [matplotlib.colors.to_rgb(color_mapping[label]) for label in labels]
        label_colors = ["rgb" + str(color) for color in label_colors]
        links["link_c"] = links["link_c"].apply(lambda x: f'rgba({x[0]*255}, {x[1]*255}, {x[2]*255}, {x[3]})')

        # Define a Plotly Sankey diagram
        fig = go.Figure( 
            data=[
                go.Sankey(
                    valuesuffix=value_suffix,
                    node=dict(label=labels, color=label_colors),
                    link=dict(
                        source=links["source"].map(lambda x: labels.tolist().index(x)),
                        target=links["target"].map(lambda x: labels.tolist().index(x)),
                        value=links["weight"],
                        color=links["link_c"],
                    ),
                )
            ]
        )

        # Customize plot based on earlier values
        fig.update_layout(
            #title_text = f"Sankey diagram of trade flows in the VOC for {selected_product} in {selected_year}",
            font_size=fontsize,
            font_family=fontfamily,
            height=600,
            paper_bgcolor=bgcolor,
            title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},  # Centers title
        )

        return fig, slider_marks

if __name__ == '__main__':
    app.run(jupyter_mode="external", port=8091)
