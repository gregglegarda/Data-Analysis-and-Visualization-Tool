
import folium
from PyQt5.QtWebEngineWidgets import QWebEngineView as QWebView
from PyQt5.QtCore import QUrl


class map_webview(QWebView):
    def __init__(self, file_path,data):

        super(map_webview, self).__init__()
        local_url = QUrl.fromLocalFile(file_path)
        self.load(local_url)
        self.create_map(data)
        self.show()

    def create_map(self, data):
        # MAP  - Create a Map instance
        m = folium.Map(
            location=[37.0902, -95.7129],
            tiles='cartodbdark_matter',
            # cartodbdark_matter #cartodbpositron #stamenwatercolor #stamenterrain #openstreetmap
            zoom_start=3.5,
            min_zoom=3.5,

        )

        # MARKERS  - Add the dataframe markers in the map
        for i in range(0, len(data)):
            folium.Circle(
                location=[data.iloc[i]['Start_Lat'], data.iloc[i]['Start_Lng']],
                popup=data.iloc[i]['Street'],
                radius=100,
                #stroke settings
                stroke = False,
                weight=1,
                color="Yellow",
                opacity = 0.2,
                #fill settings
                fill=True,
                fillColor="Yellow",
                fillOpacity=0.2,
            ).add_to(m)

        # Filepath to the output
        outfp = "map.html"
        # Save the map
        m.save(outfp)