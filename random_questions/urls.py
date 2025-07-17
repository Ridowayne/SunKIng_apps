from django.urls import path
from . import views

urlpatterns = [
    path('customer/questions', views.get_all_questions, name='customer_questions'),
    path('categories', views.category_functions, name='category'),
    path('questions', views.questions_functions, name='questions'),
    path('customerTable', views.customer_functions, name='customer'),
    # path('analysis', views.call_claude_assistant, name='ai-analysis'),
    path('analysis', views.upload_pdf_for_analysis, name='ai-analysis'),
    path('dataExtraction', views.play_with_gemini_using_gemini_client_for_large, name='ai-analysis'),
    path('callAudit', views.transcribe_call_with_gemini, name='call-audit'),
    path('callAudit/records', views.get_all_audit, name='call-audit-records'),
    path('audits/<uuid:uid>/', views.get_single_audit, name='call_audit_detail_page'),
    path('twiml-message/', views.twiml_message, name='twiml_message'),
    path('callAllCustomers/', views.call_call_overdue_lamp_customers, name='call_customer'),
    path('robocalls', views.call_call_overdue_lamp_customers_with_twilio, name='robocall'),
    path("ait_call", views.call_overdue_customers_with_ait, name='ait_call'),
    path("voice/response/", views.ait_voice_response, name='ait_voice_response'),


]