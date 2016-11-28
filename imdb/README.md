Build SQL database with with IMDB online database

Required Package
* imdbpy
* SQLObject
* postgres 

Steps
1. Download data from IMDB ftp server 
2. Create a DB in postgres call imdb
3. run imdbpy2sql.py -d /Users/qianyu/Documents/IMDB/data/ -u 'postgresql://localhost/imdb' if using a local machine
4. run imdbpy2sql.py -d /Users/qianyu/Documents/IMDB/data/ -u 'postgre://postgre:everything@localhost:5432/imdb' if using a AWS Instance

Program to Query IMDB data and generate .csv table using RESTful interface
1. Search movies by title (search_title.py) 
2. Query a movie by imdb ID
3. Create table by using a list of imdb IDs

