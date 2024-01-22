import streamlit as st
import os
from langchain_openai import ChatOpenAI
from utils.text_extraction_and_preprocessing import get_pdf_text,text_splitting
from utils.text_summarization import  num_tokens_from_string,text_summarization
from utils.category_extraction import category_extraction
import json
from dotenv import load_dotenv
load_dotenv()

#setting up the  model
model_name='gpt-3.5-turbo'
llm=ChatOpenAI(model_name=model_name,temperature=0,openai_api_key=os.getenv("OPENAI_API_KEY"))
st.set_page_config(page_title="Product categorization")
st.title("Product Categorization")
pdf_doc = st.file_uploader("Choose a PDF file", type="pdf")

#defining the product list
product_list=["Laptops","Tablets","Smartphones","Headphones","Speakers","Refrigerators","Washing Machines",
    "Dishwashers","Electric Cars","Motorcycle Helmets","Brake Pads","Engine Oil","Antibiotics","Pain Relievers",
    "Antacids","Frozen Meals","Paint strippers","Soft Drinks","Chocolate","Sunscreen","Lipstick",
    "Shampoo","Perfumes","Socks","Sneakers","Handbags","Industrial Gases","Adhesives","Cleaning Chemicals",
    "Educational Toys","Card Games","Virtual Reality Games","X-ray Machines","Defibrillators",
    "Blood Pressure Monitors","Wheat","Poultry","Herbicides","Insecticides","Fertilizer Spreader",
    "Gaming Consoles","Action Figures","Strategy Games","Drones","Fitness Trackers","Smartwatches",
    "Compact Cameras","Microwaves","Coffee Makers","Toaster Ovens","Convertible Cars","Scooters",
    "Transmission Fluid","Air Filters","Vitamins","Allergy Medications","Vaccines for Children",
    "Organic Foods","Energy Drinks","Nuts and Seeds","Facial Cleansers""Hair Conditioners",
    "Hats","Sandals","Watches","Plastic Resins","Paints and Coatings","Pesticides","Soft Toys",
    "Family Board Games","Simulation Games","3D Printers","Surgical Masks","Heart Monitors","Stethoscopes",
    "Dairy Products","Organic Fertilizers","GPS Devices","Home Security Systems",
    "Robot Vacuum Cleaners","Cordless Drills","Solar Panels","Water Heaters","Electric Bicycles",
    "Hair Dyes","Gloves","Boots","Backpacks"]


if pdf_doc is not None:
    text=get_pdf_text(pdf_doc) #extracting text from pdf
    if len(text)!=0:
        with st.spinner("Processing..."):
         
        
            docs=text_splitting(text) #creating chunks
            tokens=num_tokens_from_string(text,model_name)
            print("Number of tokens before summarization:",tokens)
            summary=text_summarization(docs,llm) #summarizing the text
            tokens_1=num_tokens_from_string(summary,model_name)
            print("Number of tokens after summarization:",tokens_1)
            print(summary)

            res=category_extraction(llm,product_list,summary) #extracting product categories 
        st.subheader("Categories")
        st.write(json.loads(res)) 
    else:
        st.write("No text found")  
    


        


