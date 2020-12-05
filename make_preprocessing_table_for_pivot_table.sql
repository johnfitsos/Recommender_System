-- Keep useful columns 

select rd.users_id, rd.rating, bd.title
from ratings_data as rd
inner join books_data as bd
on rd.isbn=bd.isbn
group by bd.title
order by rd.users_id
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/Preprocessing_columns.csv'
FIELDS TERMINATED BY ';'
LINES TERMINATED BY '\n';

-- check the number of unique user-id and of uniques book-titles

select count(distinct(rd.users_id)), count(distinct(bd.title))
from ratings_data as rd
inner join books_data as bd
on rd.isbn=bd.isbn;

