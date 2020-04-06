
import folium
from PyQt5.QtWebEngineWidgets import QWebEngineView as QWebView
from PyQt5.QtCore import QUrl
from folium.plugins import FastMarkerCluster


class map_webview(QWebView):
    def __init__(self, file_path,data):

        super(map_webview, self).__init__()
        local_url = QUrl.fromLocalFile(file_path)
        self.load(local_url)
        self.create_map(data)
        self.show()

    def create_map(self, data):

        try:
            len(data)
            if len(data) >= 5000:

            # ============================== FAST CLUSTER ==============================
                print("Creating map...")
                m2 = folium.Map(
                    location=[37.0902, -95.7129],
                    tiles='cartodbdark_matter',
                    # cartodbdark_matter #cartodbpositron #stamenwatercolor #stamenterrain #openstreetmap
                    zoom_start=3.5,
                    min_zoom=3.5,

                )
                #
                callback = ('''
                        function (row) {
                        var circle = L.circle(new L.LatLng(row[0], row[1]), {color: 'yellow', radius: 100});
                        return circle};
                                        ''')

                m2.add_child(FastMarkerCluster(data[['Start_Lat', 'Start_Lng']].values.tolist()  ,callback=callback))



                m2.save("map.html")
                #m2.save("mapcluster.html")
            else:
                # ============================== CREATE CIRCLE AND CLUSTER MAPS ==============================
                # MAP  - Create a Map instance
                print("Creating map...")
                m = folium.Map(
                    location=[37.0902, -95.7129],
                    tiles='cartodbdark_matter',
                    # cartodbdark_matter #cartodbpositron #stamenwatercolor #stamenterrain #openstreetmap
                    zoom_start=3.5,
                    min_zoom=3.5,
                )

                for lat, lan, description, street, city, state, zipcode in zip(data['Start_Lat'], data['Start_Lng'], data['Description'], data['Street'], data['City'], data['State'], data['Zipcode']):
                    print(description,type(description),street, type(street), city, type(city),state, type(state),zipcode, type(zipcode))
                    # ============================== CIRCLE ==============================
                    # CIRCLE MARKERS  - Add the dataframe markers in the map
                    folium.Circle(
                        #location=[data.iloc[i]['Start_Lat'], data.iloc[i]['Start_Lng']],
                        #popup=data.iloc[i]['Street'] + ", " + data.iloc[i]['City'] + ", " + data.iloc[i]['State'] + ", " + data.iloc[i]['Zipcode'],
                        location=[lat, lan],
                        popup= description + ", " + street + ", " + city + ", " + state + ", " + zipcode,
                        radius=10,
                        #stroke settings
                        stroke = False,
                        weight=1,
                        color="Yellow",
                        opacity = 0.2,
                        #fill settings
                        fill=True,
                        fill_color="Yellow",
                        fill_opacity= 0.8,
                    ).add_to(m)
                # Save the map
                m.save("map.html")
        except:
            print("Initializing map...")
            m3 = folium.Map(
                location=[37.0902, -95.7129],
                tiles='cartodbdark_matter',
                # cartodbdark_matter #cartodbpositron #stamenwatercolor #stamenterrain #openstreetmap
                zoom_start=3.5,
                min_zoom=3.5,

            )
            m3.save("map.html")
        print("Map created")

