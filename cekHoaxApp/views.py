# from concurrent.futures import process
from django.shortcuts import render
from cekHoaxApp import predict
from cekHoaxApp.prosesHoax import *

# Create your views here.
def index(request):
    konteks = {
        'title': 'cekHoax',
        'subtitle': '',
        'active': 'cekHoax',
    }
    return render(request, 'index.html', konteks)

def scanText(request):

    if request.method == "POST":
        text = request.POST.get('judul')
        hasil = predict.predictor(text)
        print(hasil)
        konteks = {
            'title': 'cekJudul',
            'subtitle': '',
            'active': 'cekHoax',
            'result': hasil
        }
        return render(request, 'judul.html', konteks)

def scanNarasi(request):

    narasi = request.POST['narasi']
    hasil = proses(narasi, 'narasi')

    konteks = {
        'title': 'cekNarasi',
        'subtitle': '',
        'active': 'cekHoax',
        'hasil': hasil
    }
    return render(request, 'narasi.html', konteks)