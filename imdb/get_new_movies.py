from imdb import IMDb
import StringIO
import csv

out = StringIO.StringIO()
f1 = open('movie_data.csv', 'w')
ia = IMDb('http')

movie1 = ia.get_movie('3748528')
movie2 = ia.get_movie('1211837')
movie3 = ia.get_movie('3183660')
movie4 = ia.get_movie('3521164')
movie5 = ia.get_movie('1619029')
movie6 = ia.get_movie('3783958')
movie7 = ia.get_movie('4682786')
movie8 = ia.get_movie('1355644')
movie9 = ia.get_movie('2094766')
movie10 = ia.get_movie('3470600')
movies = (movie1, movie2, movie3, movie4, movie5, movie6, movie7, movie8, movie9, movie10)


cw = csv.writer(out)
header = ("Title", "Cast", "Director", "Genre", "language", "Plot_Summary", "imdb_URL")
cw.writerow(header)
print header

for movie in movies:

    imdbURL = ia.get_imdbURL(movie)
    if not imdbURL:
        imdbURL = 'NA'

    genres = movie.get('genres')
    if not genres:
        genres = 'NA'

    director = movie.get('director')
    if not director:
        director = 'NA'
    else:
        director = director[0]

    lang = movie.get('lang')
    if not lang:
        lang = 'NA'

    plot = movie.get('plot')
    if not plot:
        plot = 'NA'

    cast = movie.get('cast')
    if not cast:
        cast = 'NA'
    else:
        cast = cast[:5]
        main_cast = [name['name'] for name in cast]

    data = (movie['title'], main_cast, director, genres, lang, plot, imdbURL)
    cw.writerow(data)
    print data

f1.write(out.getvalue())
