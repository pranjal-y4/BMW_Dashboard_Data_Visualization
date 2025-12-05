import os                       
import pandas as pd             # data process and analysis
import numpy as np              # NumPy for preprovess
from dash import Dash, html, dcc, Input, Output   
import plotly.express as px     # High-level Plotly API for quick charts
import plotly.graph_objects as go  # Low-level Plotly API for custom charts
import math                     #   i mightr require but again -------

# Get the dir path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Full path to the CSV
csv_det = os.path.join(BASE_DIR, "BMW sales data (2010-2024) (1).csv")

# DataFrame
df = pd.read_csv(csv_det)

# Remove white space
df.columns = df.columns.str.strip()

# Remove white space and conver them to string type
for col in ["Region", "Color", "Fuel_Type", "Transmission", "Model", "Sales_Classification"]:
    if col in df.columns:                      # col process if exists
        df[col] = df[col].astype(str).str.strip()

# this is what we used on slider that is year
df["Year"] = df["Year"].astype(int)

# Numeric to numeric dtype
for col in ["Engine_Size_L", "Mileage_KM", "Price_USD", "Sales_Volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")  # errors="coerce" turns invalid values into NaN

# Compute min & max prices for price range 
l_price, h_price = df["Price_USD"].min(), df["Price_USD"].max()

# 6 spaced price interval bw min and max price
b_price = pd.interval_range(start=l_price, end=h_price, periods=6)


# Min and max years present in the dataset
min_year_vis, max_year_vis = df["Year"].min(), df["Year"].max()

# sorted list regions (drop NaNs first)
regions_all_of_them = sorted(df["Region"].dropna().unique())

# unique list of furel typ
fuels = sorted(df["Fuel_Type"].dropna().unique())

# same cra model
ALL_MODELS = sorted(df["Model"].dropna().unique())

# Estimated revenue = price * volume
df["Estimated_Revenue"] = df["Price_USD"] * df["Sales_Volume"]

# Find total sales volume and revenue for each model, sort by vol and pick 3
top_models_BMW = (
    df.groupby("Model")[["Sales_Volume", "Estimated_Revenue"]]  # group data by model
      .sum()                                                   # sum volume + revenue
      .reset_index()                                           # turn index back into columns
      .sort_values("Sales_Volume", ascending=False)            # sort descending - volume
      .head(3)                                                 # keep top 3
)

# Choose a default model for the depreciation curve:- first model among top-selling 3, first model in the full list, if nothing exists at all, keep None
DEFAULT_DEPR_MODEL = (
    top_models_BMW.iloc[0]["Model"]
    if not top_models_BMW.empty
    else (ALL_MODELS[0] if ALL_MODELS else None)
)

# Hard-coded approximate coordinates (lat, lon) for major regions
region_cooords = {
    "North America": (40.0, -100.0),   # Approx center of NA
    "South America": (-15.0, -60.0),   # Approx center of SA
    "Europe": (54.0, 15.0),            # Rough central value Europe
    "Asia": (34.0, 100.0),             # Rough center value Asia
    "Middle East": (26.0, 45.0),       # Approx Middle East
    "Africa": (0.0, 20.0),             # Approx center value Africa
}

# Coordinates for BMW HQ  as Munich, Germany - this are reference point 
BMW_HQ_LAT, BMW_HQ_LON = 48.1351, 11.5820

# aesthetics and color
BG_PAGE = "#020617"      # page base color
BG_CARD = "#0b1120"      # cell background (for charts)
FG_TEXT = "#F5F7FA"      # main text color
ACCENT = "#1d4ed8"       # primary accent (charts, markers)
ACCENT_2 = "#E0E0E0"     # secondary accent (labels etc.)

# Background gradient string
page_color_or_grad = "linear-gradient(135deg, #020617 0%, #020617 25%, #0b1120 55%, #1d3557 80%, #020617 100%)"

#helper functions

# lets apply dark theme layout to any pltoly figure that we will make
def style_fig(fig, title=None, height=None):
    # Update the figure's layout with common styling attributes
    fig.update_layout(
        template="plotly_dark",           # using Plotly's dark theme template as a base
        paper_bgcolor=BG_CARD,            
        plot_bgcolor=BG_CARD,             
        font_color=FG_TEXT,               # using default font color
        font=dict(size=11),               # using font size
       #titkle changs
       title=dict(
            text=title or "",             
            x=0.01, xanchor="left",       
            y=0.98, yanchor="top",        
            font=dict(size=12)          
        ),
        margin=dict(l=8, r=8, t=26, b=12) #margins around plot
    )
    # If a height is provided, apply it to the figure
    if height:
        fig.update_layout(height=height)
    return fig

#filter global df based on current UI
def filter_data(year_range, regions, fuels_list, models):
    d = df.copy()  # start from the full data

    # Filter by year range
    if year_range:
        d = d[(d["Year"] >= year_range[0]) & (d["Year"] <= year_range[1])]

    # selected regions
    if regions:
        d = d[d["Region"].isin(regions)]

    # selected fuel
    if fuels_list:
        d = d[d["Fuel_Type"].isin(fuels_list)]

    #selected models
    if models:
        d = d[d["Model"].isin(models)]

    return d

#build empty fig with a cemtered message when filter data is empty
def empty_figure(msg="No data for current filters"):
    fig = go.Figure()  # blank fig

    # Single text annotation in the middle of the figure
    fig.add_annotation(
        text=msg,             # message to display
        x=0.5, y=0.5,         # center of the figure (paper coords)
        xref="paper", yref="paper",
        showarrow=False,      # no arrow, just text
        font=dict(size=14, color=FG_TEXT)
    )

    # Hide both axes (no ticks or grid)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    # Apply standard styling
    return style_fig(fig, title=None)

#great circle diatance using haversiie
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers now why we need this because of the formula imp

    # Convert all lat and lon from deg to rad
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    # Haversine formula - central angle
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # dist = radius * central angle
    return R * c

#make a  a map - BMW HQ diatance, , info block per region, scale bar for legend
def geo_map_2(d, height=260):
    #filter is dataset is empty then we will show placeholder
    if d.empty:
        return empty_figure()

    # total sales volume by region
    agg = d.groupby("Region")["Sales_Volume"].sum().reset_index()

    # approx lat/lon for each region using region_cooords
    lats, lons = zip(*[region_cooords.get(r, (0, 0)) for r in agg["Region"]])
    agg["lat"], agg["lon"] = lats, lons

    #great-circle distance from BMW HQ to region centroid
    agg["dist_km"] = [
        haversine(BMW_HQ_LAT, BMW_HQ_LON, la, lo) for la, lo in zip(agg["lat"], agg["lon"])
    ]

    #min and max sales volumes
    min_sv, max_sv = agg["Sales_Volume"].min(), agg["Sales_Volume"].max()

    # size series
    agg["size"] = 10 + 30 * (agg["Sales_Volume"] - min_sv) / (max_sv - min_sv + 1e-5)

    # Initialize an empty figure for Scattergeo
    fig = go.Figure()

    # Add marker
    for _, row in agg.iterrows():
        fig.add_trace(go.Scattergeo(
            lat=[row["lat"]],        # lat
            lon=[row["lon"]],        # long
            text=(                   # hover feature is 
                f"{row['Region']}<br>"
                f"Sales: {row['Sales_Volume']:,}<br>"
                f"Distance from HQ: {row['dist_km']:.0f} km"
            ),
            hoverinfo="text",        # text 
            mode="markers",          # Scattergeo for marker
            showlegend=False,        # do not show in the legend
            marker=dict(
                size=row["size"],    # marker size scaled by vol
                color=ACCENT,        # marker color
                opacity=0.95         # ttransparent
            )
        ))

    #Base world map: natural Earth projection, static scope
    fig.update_geos(
        scope="world",               
        projection_type="natural earth",  
        showland=True,               
        landcolor="#99865B",         
        oceancolor=BG_CARD,          
        showcountries=True,         
        countrycolor="#0f172a",      
        lakecolor=BG_CARD,           
        showocean=True               
    )

    # same style throughout
    fig = style_fig(fig, "Global Sales Presence & Distance to HQ", height)

    # reduce margins in this
    fig.update_layout(
        margin=dict(l=0, r=0, t=26, b=0),
        showlegend=False
    )

    # text summarisinng metric per rgion
    dist_lines = [
        f"{row.Region}: {row.dist_km:.0f} km"
        for row in agg.itertuples(index=False)
    ]

    # make into 1 block
    dist_block = (
        "<b>Distance metric</b><br>"
        "Great-circle distance from BMW HQ (Munich).<br>"
        "Rule of thumb: 1° latitude ≈ 111 km.<br><br>"
        + "<br>".join(dist_lines)
    )

    # Annotation box in the bottom-left of the figure
    fig.add_annotation(
        text=dist_block,
        x=0.02, y=0.02,               # 2% from left, 2% from bottom
        xref="paper", yref="paper",   # coordinates relativ
        showarrow=False,              
        align="left",                 
        bordercolor="#1f2937",        
        borderwidth=0.5,              
        borderpad=4,                  
        bgcolor="rgba(15,23,42,0.85)",
        font=dict(size=9, color=FG_TEXT)
    )

    # Scale bar representing approximately 2000 km 

    # distance in km that the scale bar
    distance_km = 2000

    # reference lat near the bottom map
    lat_ref = -50

    #km per degree of long at this lat: 111.32 * cos(phi)
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat_ref))

    # Convert the degrees
    deg_lon_span = distance_km / km_per_deg_lon

    #center horizontally
    lon_center = 0
    lon_start = lon_center - deg_lon_span / 2
    lon_end = lon_center + deg_lon_span / 2

    # Add the line representing the scale bar on the map
    fig.add_trace(go.Scattergeo(
        lon=[lon_start, lon_end],   
        lat=[lat_ref, lat_ref],      
        mode="lines",               
        showlegend=False,          
        line=dict(width=3, color="#ffffff")  
    ))

    # Add a label for the scale bar
    fig.add_annotation(
        text="≈ 2000 km",             # label text
        x=0.5, y=0.05,                # centered horizontally near bottom
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=9, color=FG_TEXT),
        bgcolor="rgba(15,23,42,0.8)", # dark bg behind label
        bordercolor="#1f2937",
        borderwidth=0.5,
        borderpad=3
    )

    return fig

#animated graph
def fuel_anim(d, height=260):
    #filtering show empty figure if nothing
    if d.empty:
        return empty_figure("No data for animation")

    #total sales volume by Year and Fuel_Type
    agg = d.groupby(["Year", "Fuel_Type"])["Sales_Volume"].sum().reset_index()

    #min and max of sales volume for y-axis scaling
    y_max = agg["Sales_Volume"].max()
    y_min = agg["Sales_Volume"].min()

    # Define lower and up bounds for y-axis
    lower = max(0, y_min * 0.8)
    upper = y_max * 1.1

    # animated bar chart using plotly express
    fig = px.bar(
        agg,
        x="Fuel_Type",                 
        y="Sales_Volume",             
        color="Fuel_Type",             
        animation_frame="Year",        # animation frames by year
        range_y=[lower, upper]        
    )

    # labels on each bar and hide legend
    for tr in fig.data:
        tr.update(
            texttemplate="%{y:,}",      
            textposition="outside",     # labels above the bars
            cliponaxis=False,           
            showlegend=False            
        )

    #y-axis and enforce integer-like ticks with thousands separator
    fig.update_yaxes(title="Sales Volume", tickformat=",", rangemode="tozero")

    #x-axis and allow margins so labels don't get cut off
    fig.update_xaxes(title="Fuel Type", automargin=True)

    # hide
    fig.update_layout(showlegend=False)

    # Apply standard styling and return
    return style_fig(fig, "Animated Fuel Mix Over Time", height)

#subburst chart 
def color_sunburst(d, height=260):
    #no data, return empty 
    if d.empty:
        return empty_figure()

    #Sales volume by Color and Model
    agg = d.groupby(["Color", "Model"])["Sales_Volume"].sum().reset_index()

    #color
    color_map = {
        "Red": "#b22222",
        "Blue": "#1f77b4",
        "Black": "#000000",
        "White": "#ffffff",
        "Silver": "#C0C0C0",
        "Grey": "#808080",
        "Gray": "#808080"
    }

    # Build the sunburst with this
    fig = px.sunburst(
        agg,
        path=["Color", "Model"],
        values="Sales_Volume",
        color="Color",
        color_discrete_map=color_map
    )

    #color segment for each segment because of not being readable
    if fig.data:
        trace = fig.data[0]  
        label_colors = []    
        for lbl in trace.labels:
            l = str(lbl).strip().lower()
            if l == "black":
                label_colors.append("#ffffff")  # white text for black 
            elif l == "white":
                label_colors.append("#000000")  # black text for white 
            else:
                label_colors.append(FG_TEXT)    
        trace.textfont = dict(color=label_colors, size=10)

    fig.update_traces(insidetextorientation="radial")

    fig = style_fig(fig, "Color → Model", height)
    fig.update_layout(
        margin=dict(l=0, r=0, t=24, b=0),     
        uniformtext=dict(minsize=9, mode="hide")  # hide labels small
    )

    return fig

#multi facteed apike
def trans_fuel(d, height=260):

    if d.empty:
        return empty_figure()

    # Sales volume by Year, Fuel_Type, and Transmission
    agg = d.groupby(["Year", "Fuel_Type", "Transmission"])["Sales_Volume"].sum().reset_index()

    # Build a line chart with x=Year, y=Sales_Volume, color by Fuel_Type, facet by Transmission
    fig = px.line(
        agg,
        x="Year",
        y="Sales_Volume",
        color="Fuel_Type",
        facet_row="Transmission",
        facet_row_spacing=0.03,
        markers=True              
    )

    fig.update_traces(line=dict(width=2), marker=dict(size=5))

    # Label y-axis
    fig.update_yaxes(title_text="Sales Volume", tickformat=",")

    # base stye
    fig = style_fig(fig, "Sales Trend by Fuel & Transmission", height)

    # vertical legend
    fig.update_layout(
        legend=dict(
            orientation="v",                 
            x=1.02,                         
            xanchor="left",
            y=0.5,                           
            yanchor="middle",
            font=dict(size=9),
            bgcolor="rgba(15,23,42,0.85)"    # semi-transparent dark background
        ),
        margin=dict(l=0, r=0, t=24, b=0)
    )

    # remove "Transmission="
    for ann in fig.layout.annotations:
        if "Transmission=" in ann.text:
            ann.text = ann.text.replace("Transmission=", "").strip()

    return fig

# bar chart
def model_price_bar(d, height=200):

    if d.empty:
        return empty_figure()

    # Group by model and compute average price
    agg = (
        d.groupby("Model")["Price_USD"]
         .mean()
         .reset_index()
         .sort_values("Price_USD", ascending=False)
    )

    #models vs average price
    fig = px.bar(agg, x="Model", y="Price_USD")

    fig.update_traces(width=0.5)

    buffer = (agg["Price_USD"].max() - agg["Price_USD"].min()) * 0.15
    fig.update_yaxes(
        range=[max(0, agg["Price_USD"].min() - buffer), agg["Price_USD"].max() + buffer],
        title="Average Price (USD)",
        tickformat=","
    )

    #X-axis 
    fig.update_xaxes(title="Model", tickangle=-45)

    return style_fig(fig, "Average Price by Model", height)

#depreciation curve
def depreciation(d, selected_model, height=200):

    if d.empty:
        return empty_figure()

    #filter and drop rows with missing mileage/price
    dm = d[d["Model"] == selected_model].dropna(subset=["Mileage_KM", "Price_USD"])

    #least 5 rows for a meaningful curve
    if len(dm) < 5:
        return empty_figure("Not enough data for depreciation curve")

    # Bin mileage into quantiles
    dm["Mileage_bin"] = pd.qcut(dm["Mileage_KM"], q=min(10, len(dm) // 2), duplicates="drop")

    # Aggregate within each mileage bin
    agg = dm.groupby("Mileage_bin").agg(
        avg_mileage=("Mileage_KM", "mean"),
        avg_price=("Price_USD", "mean"),
        count=("Price_USD", "size")
    ).reset_index()

    fig = go.Figure(go.Scatter(
        x=agg["avg_mileage"],           # x-axis
        y=agg["avg_price"],             # y-axis
        mode="lines+markers",           # draw lines and markers
        line=dict(color=ACCENT, width=2.3),
        marker=dict(size=6),
        text=agg["count"],              
        hovertemplate=(
            "Avg mileage: %{x:.0f} km<br>"
            "Avg price: %{y:.0f} USD<br>"
            "Cars in band: %{text}<extra></extra>"
        )
    ))

    fig.update_xaxes(title="Mileage (km)")
    fig.update_yaxes(title="Average Price (USD)", tickformat=",")

    return style_fig(fig, "Depreciation Curve", height)

app = Dash(__name__)

# browser title
app.title = "BMW Sales Analysis - Full Width"

# Define the layout of the app
app.layout = html.Div([
    # header
    html.Div([
        html.Div([
            # BMW logo (if asset exists)
            html.Img(
                src=app.get_asset_url("bmw_logo.png") if os.path.exists(os.path.join(BASE_DIR, "assets", "bmw_logo.png")) else "",
                style={
                    "height": "44px", "width": "44px", "marginRight": "10px", "borderRadius": "50%", "boxShadow": "0 0 10px rgba(29,78,216,0.8)"
                }
            ),
            # Title and subtitle
            html.Div([
                html.H1(
                    "BMW Sales Analysis Dashboard",
                    style={"margin": 0, "fontSize": "20px", "color": FG_TEXT}
                ),
                html.P(
                    "Global footprint, fuel mix and customer preferences.",
                    style={"margin": 0, "color": "#AAB2C5", "fontSize": "11px"}
                )
            ], style={
                "flex": 1,
                "display": "flex",
                "flexDirection": "column",
                "justifyContent": "center"
            })
        ], style={
            "display": "flex", "alignItems": "center", "padding": "6px 10px", "background": "rgba(2,6,23,0.9)", "borderBottom": "1px solid #1C2330"
        })
    ]),

    # Filter 
    html.Div([
        # Year range slider
        html.Div([
            html.Label("Year range", style={"color": ACCENT_2, "fontSize": "11px"}),
            dcc.RangeSlider(
                id="year_slider",
                min=min_year_vis,          # earliest year in data
                max=max_year_vis,          # latest year in data
                step=1,                    # step by 1 year
                value=[min_year_vis, max_year_vis],  
                marks={                     
                    y: str(y)
                    for y in range(min_year_vis, max_year_vis + 1)
                    if y == min_year_vis or y == max_year_vis or y % 3 == 0
                }
            )
        ], style={"flex": 2, "padding": "6px 8px"}),

        # Region filter (multi-select dropdown)
        html.Div([
            html.Label("Region (empty = all)", style={"color": ACCENT_2, "fontSize": "11px"}),
            dcc.Dropdown(
                id="region_filter",
                options=[{"label": r, "value": r} for r in regions_all_of_them],
                value=[],                  # default: no specific selection (means all)
                multi=True,
                placeholder="All regions"
            )
        ], style={"flex": 1, "padding": "6px 8px"}),

        # Fuel type filter
        html.Div([
            html.Label("Fuel type (empty = all)", style={"color": ACCENT_2, "fontSize": "11px"}),
            dcc.Dropdown(
                id="fuel_filter",
                options=[{"label": f, "value": f} for f in fuels],
                value=[],                  # default: all fuel types
                multi=True,
                placeholder="All fuel types"
            )
        ], style={"flex": 1, "padding": "6px 8px"}),

        # Model filter
        html.Div([
            html.Label("Models (empty = all)", style={"color": ACCENT_2, "fontSize": "11px"}),
            dcc.Dropdown(
                id="model_filter",
                options=[{"label": m, "value": m} for m in ALL_MODELS],
                value=[],                  # default: all models
                multi=True,
                placeholder="All models"
            )
        ], style={"flex": 1.2, "padding": "6px 8px"})
    ], style={
        "display": "flex", "flexWrap": "wrap", "padding": "6px 8px", "background": "rgba(11,17,32,0.95)", "borderBottom": "1px solid #1C2330"
    }),

    # Row 1: Fuel animation | Global map
    html.Div([
        html.Div(
            dcc.Graph(id="fuel_animation", style={"height": "260px"}),
            style={
                "flex": 1.0, "margin": "4px", "backgroundColor": BG_CARD, "borderRadius": "8px"
            }
        ),
        html.Div(
            dcc.Graph(id="region_geo_map", style={"height": "260px"}),
            style={
                "flex": 1.0, "margin": "4px", "backgroundColor": BG_CARD, "borderRadius": "8px"
            }
        )
    ], style={
        "display": "flex", "flexWrap": "wrap", "padding": "4px 6px"
    }),

    # Row 2: Sunburst | Spike chart 
    html.Div([
        html.Div(
            dcc.Graph(id="color_dna_sunburst", style={"height": "260px", "margin": "0"}),
            style={
                "flex": 0.85,              # narrower width than spike chart
                "margin": "0 2px 0 6px",
                "backgroundColor": BG_CARD,
                "borderRadius": "8px",
                "padding": "0"             # no inner padding
            }
        ),
        html.Div(
            dcc.Graph(id="transmission_fuel_trend", style={"height": "260px", "margin": "0"}),
            style={
                "flex": 1.15,              # takes more width for spikes and legend
                "margin": "0 6px 0 2px",
                "backgroundColor": BG_CARD,
                "borderRadius": "8px",
                "padding": "0"
            }
        )
    ], style={
        "display": "flex", "flexWrap": "wrap", "padding": "0 4px", "margin": "2px 0"
    }),

    html.Div([
        html.Div(
            dcc.Graph(id="model_price_bar", style={"height": "200px"}),
            style={
                "flex": 1,
                "margin": "4px",
                "backgroundColor": BG_CARD,
                "borderRadius": "8px"
            }
        ),
        html.Div([
            html.Div([
                html.Span("Depreciation Curve – ", style={"color": FG_TEXT, "fontSize": "12px"}),
                # Dropdown for selecting model for depreciation curve
                dcc.Dropdown(
                    id="depr_model",
                    options=[{"label": m, "value": m} for m in ALL_MODELS],
                    value=DEFAULT_DEPR_MODEL,     # default: top-selling model
                    style={"width": "220px"}
                )
            ], style={
                "display": "flex",
                "alignItems": "center",
                "padding": "4px 8px 0 8px"
            }),
            # Depreciation curve graph
            dcc.Graph(id="depr_curve", style={"height": "180px", "padding": "0px 4px 4px 4px"})
        ], style={
            "flex": 1,
            "margin": "4px",
            "backgroundColor": BG_CARD,
            "borderRadius": "8px"
        })
    ], style={
        "display": "flex",
        "flexWrap": "wrap",
        "padding": "4px 6px 16px 6px"
    })

], style={
    "width": "100%",                  
    "minHeight": "100vh",             
    "margin": "0",                    
    "padding": "0 0 8px 0",           
    "background": page_color_or_grad, 
    "fontFamily": "system-ui, sans-serif"
})


@app.callback(
    Output("fuel_animation", "figure"),          
    Output("region_geo_map", "figure"),          
    Output("color_dna_sunburst", "figure"),      
    Output("transmission_fuel_trend", "figure"), 
    Output("model_price_bar", "figure"),         
    Input("year_slider", "value"),               
    Input("region_filter", "value"),             
    Input("fuel_filter", "value"),               
    Input("model_filter", "value")               
)

#min callback to update
def update_main_figs(year_range, regions, fuels_list, models):
    d = filter_data(year_range, regions, fuels_list, models)

    return (
        fuel_anim(d),
        geo_map_2(d),
        color_sunburst(d),
        trans_fuel(d),
        model_price_bar(d)
    )

@app.callback(
    Output("depr_curve", "figure"),              
    Input("year_slider", "value"),               
    Input("region_filter", "value"),
    Input("fuel_filter", "value"),
    Input("model_filter", "value"),
    Input("depr_model", "value")                 
)

#call back
def update_depr_curve(year_range, regions, fuels_list, models, depr_model):

    d = filter_data(year_range, regions, fuels_list, models)

    return depreciation(d, depr_model)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8055)
