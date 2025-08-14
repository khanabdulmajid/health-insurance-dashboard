# ml_dashboard/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_home, name='dashboard_home') ,
    # path('dashboard',views.render_dashboard,name = 'dashboard')
]
