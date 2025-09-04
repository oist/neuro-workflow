# workflow_backend
workflow system backend (django & jupyter-lab)

1.Startup procedure

> cd workflow_backend

> docker-compose build

> docker-compose up


2.If you can't start, try the following.

2-1 Delete a table in the DB
> docker-compose exec -it db bash
db> psql -U postgres -d django_
psql> DROP TABLE IF EXISTS box_pythonfile CASCADE;
psql> DROP TABLE IF EXISTS node_definitions CASCADE; 

2-2 Resetting the migration file
> docker-compose exec -it web bash
web> python sjango-project/manage.py makemigrations
web> python sjango-project/manage.py migrate
