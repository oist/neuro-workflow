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
psql> DELETE FROM django_migrations WHERE app='box';

2-2 Resetting the migration file
> docker-compose down -v
> docker-compose up
> docker-compose exec -it web bash
web> python django-project/manage.py makemigrations
web> python django-project/manage.py migrate

2-3 Build NEST simulator enabled JupyterLab user server image
> cd django-project/neuroworkflow
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

