from django.db import models
import uuid
import random


class BaseModel(models.Model):
    uid=models.UUIDField(primary_key=True,default=uuid.uuid4,editable=False)
    created_at=models.DateField(auto_now_add=True)
    updated_at= models.DateField(auto_now_add=True)

    class Meta:
        abstract=True

class Category(BaseModel):
    category_name=models.CharField(max_length=100)
    category_number = models.IntegerField(default=0)
    archived=models.BooleanField(default=False)
    country = models.CharField(max_length=200, default="Nigeria")

    def __str__(self) -> str:
        return self.category_name

class Question(BaseModel):
    category=models.ForeignKey(Category, related_name='category',on_delete=models.CASCADE)
    question=models.CharField(max_length=300)
    archived=models.BooleanField(default=False)
    
    

    def __str__(self) -> str:
        return self.question

    def get_answers(self):
        answer_objs=list(Answer.objects.filter(question = self))
        random.shuffle(answer_objs)
        data=[]

        for answer_obj in answer_objs:
            data.append({
                'answer' : answer_obj.answer,
            })

        return data

class Answer(BaseModel):
    question=models.ForeignKey(Question, related_name='question_answer',on_delete=models.CASCADE)
    answer=models.CharField(max_length=100)
    marks=models.IntegerField(default=5)

    def __str__(self) -> str:
        return self.answer
    
class Customer(BaseModel):
    name = models.CharField(max_length=200, null=False)
    total_score = models.IntegerField(default=0)
    product_category = models.CharField(max_length=330)
    country = models.CharField(max_length=200, default="Nigeria")
    area = models.CharField(max_length=200, null=False, default="Lagos")

    def __str__(self) -> str:
        return self.name
    

class CallAudit(BaseModel):
    name = models.CharField(max_length=200, null=False)
    agent_id = models.CharField(max_length=200, default="00001")
    auidited_by = models.CharField(max_length=200, null=False)
    total_score = models.IntegerField(default=0)
    foundation_Skill_score = models.IntegerField(default=0)
    commuication_skill_score = models.IntegerField(default=0)
    probing_skill_score = models.IntegerField(default=0)
    negotiation_skill_score = models.IntegerField(default=0)
    closing_skill_score = models.IntegerField(default=0)
    agent_disposition = models.CharField(max_length=200, null=False)
    call_language = models.CharField(max_length=200, null=False)
    call_date = models.DateField(auto_now_add=True)
    justification = models.TextField(null=True)

    def __str__(self) -> str:
        return self.name