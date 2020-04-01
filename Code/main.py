
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import pandas as pd


#########-------------------------------------- DATA FRAME -------------------------------------- #########
df = pd.read_csv("US_Accidents_Dec19.csv")
#df = pd.read_csv("https://www.kaggle.com/sobhanmoosavi/us-accidents#US_Accidents_Dec19.csv")
#df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_february_us_airport_traffic.csv')
df = df.sample(100)
print(df.columns)
del df['Source']
print(df.columns)
print(df.head(10))



df['text'] = df['Street'] +', '+ df['City']+ ', '+ df['County']+ ' ' + df['State']+ ' '+ df['Zipcode'].astype(str)
scl = [0,"rgb(150,0,90)"],[0.125,"rgb(0, 0, 200)"],[0.25,"rgb(0, 25, 255)"],\
[0.375,"rgb(0, 152, 255)"],[0.5,"rgb(44, 255, 150)"],[0.625,"rgb(151, 255, 0)"],\
[0.75,"rgb(255, 234, 0)"],[0.875,"rgb(255, 111, 0)"],[1,"rgb(255, 0, 0)"]


fig = go.Figure(data=go.Scattergeo(
        lon = df['Start_Lng'],
        lat = df['Start_Lat'],
        text = df['text'],
        mode = 'markers',
        #marker_color = df['Visibility(mi)'],
        marker = dict(
                color = df['Visibility(mi)'],
                colorscale = "Inferno",
                reversescale = True,
                opacity = 0.7,
                size = 2,
                colorbar = dict(
                    titleside = "right",
                    outlinecolor = "rgba(68, 68, 68, 0)",
                    ticks = "outside",
                    showticksuffix = "last",
                    dtick = 10
                        ))

        ))

fig.update_layout(
        title = 'USA Traffic Accidents 2019<br>(Hover for accident details)',
        geo_scope='usa',
        #height=300, margin={"r":0,"t":0,"l":0,"b":0}
    )
fig.update_geos(
        showland=True, landcolor="#222222",
        #showcoastlines=True, coastlinecolor="dimgrey",
        #showocean=True, oceancolor="dimgrey",
        #showlakes=True, lakecolor="dimgrey",
        #showrivers=True, rivercolor="dimgrey"
)
fig.show()