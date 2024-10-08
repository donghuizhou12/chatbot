# Generated by Django 5.1.1 on 2024-09-29 05:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0003_specialist'),
    ]

    operations = [
        migrations.AddField(
            model_name='specialist',
            name='experience',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='specialist',
            name='location',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AlterField(
            model_name='specialist',
            name='contact',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
