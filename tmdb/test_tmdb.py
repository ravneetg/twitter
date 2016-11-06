import tmdbsimple as tmdb
import StringIO
import csv

tmdb.API_KEY = "7ef81c799d9a02c3e8ed0e7b07092471"
out = StringIO.StringIO()
f = open('data_out.csv', 'w')

movie = tmdb.Movies(603)
response = movie.info()
print movie.title
print movie.budget
print movie.popularity
print movie.revenue

search = tmdb.Search()
reponse = search.movie(query = 'star wars')
for s in search.results:
    data = (s['title'], s['id'], s['release_date'], s['popularity'])
    cw = csv.writer(out)
    cw.writerow(data)

print out.getvalue()
f.write(out.getvalue())
