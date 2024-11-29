import langchain_helper as lch
import streamlit as st

st.title("Pets name generator")
    
user_animal = st.sidebar.selectbox("What is pet?",("cat","dog","parrot","rabbit"))

if user_animal=="cat":
    pet_color = st.sidebar.text_area(label="What color of your cat is?",max_chars=15)

if user_animal=="dog":
    pet_color = st.sidebar.text_area(label="What color of your dog is?",max_chars=15)

if user_animal=="parrot":
    pet_color = st.sidebar.text_area(label="What color of your parrot is?",max_chars=15)

if user_animal=="rabbit":
    pet_color = st.sidebar.text_area(label="What color of your rabbit is?",max_chars=15)

if pet_color:
    response = lch.generate_pet_names(user_animal,pet_color)
    st.text(response['pet_name'])

#import openai



