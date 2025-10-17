# workflow_backend
workflow system backend (django & jupyter-lab)

1.Startup procedure
1-1 Starting Docker components
> cd workflow_backend

> docker-compose build

> docker-compose up

The following components start successfully:
  web
  db
  jupyterhub

1-2 Stopping and restarting Docker components
> cd workflow_backend
> docker-compose down -v

1-3 If the above does not stop the process, kill it
Stop the db, web, and jupyterhub component processes.
> docker inspect -f '{{.State.Pid}}' "db"
1234
> sudo kill -9 1234


2.If you can't start, try the following.

2-1 Delete a table in the DB (PostgreSQL)
> cd workflow_backend
> docker-compose exec -it db bash
db> psql -U postgres -d django_
psql> DROP TABLE IF EXISTS box_pythonfile CASCADE;
psql> DELETE FROM django_migrations WHERE app='box';
psql> DELETE FROM django_migrations WHERE app='workflow';

2-2 Delete the sqlite database file
> cd workflow_backend
> rm nodes.db

2-3 Delete the django migration file
If a migration directory has been created under the "workflow_backend/" directory, delete the files in it whose names begin with zero-padded numbers such as "0001_" or "0002_".

2-4 Resetting the migration file
> cd workflow_backend
> docker-compose down -v
> docker-compose up
> docker-compose exec -it web bash
web> python django-project/manage.py makemigrations
web> python django-project/manage.py migrate

2-5 Build NEST simulator enabled JupyterLab user server image
> cd workflow_backend/django-project/neuroworkflow
> ./build-nest-image.sh

3.django management screen

3-1 Create administrative user
> docker-compose exec -it web bash
web> python manage.py createsuperuser
     ... Register django admin user

3-2 Management screen URL
http://localhost:3000/admin


4.jupyter hub

4-1 Settings
> docker-compose exec -it web bash
web> cd django-project/neuroworkflow/
web> ./build-nest-image.sh

4-2 jupyter hub URL
http://localhost:8000/
  user: user1
  password: password
