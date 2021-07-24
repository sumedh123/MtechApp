import streamlit as st
import os
import time
import cv2
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance
from bs4 import BeautifulSoup
from keras.models import load_model
st.set_page_config(page_title="Respiratory Disease Detection Tool", page_icon="AppIcon.jpg", layout='centered', initial_sidebar_state='auto')



def main():

	URL = "https://www.worldometers.info/coronavirus/#countries"
	dic = {}
	col_names = ["Total Cases","New Cases", "Total Deaths", "New Deaths", "Total Recovered"]
	html_page = requests.get(URL).text
	soup = BeautifulSoup(html_page,'lxml')
	get_table = soup.find("table", id = "main_table_countries_today")
	get_table_data = get_table.tbody.find_all("tr")

	for i in range(len(get_table_data )):
		try:
			key = get_table_data[i].find_all("a", href = True)[0].string
		except:
			key = get_table_data[i].find_all("td")[0].string
		if key == None: 
			continue
		values = [j.string for j in get_table_data[i].find_all("td")]
		dic[key] = values

	df = pd.DataFrame(dic).iloc[2:,:].T.iloc[:,:5]
	df.index_name = "country"
	df.columns = col_names


	"""A Proposal for Respiratory Disease Diagnosis powered by Deep Learning and Streamlit"""
	html_templ = """
	<div style="background-color:Aquamarine;padding:10px;">
	<h1 style="color:OrangeRed">Respiratory Disease Detection Tool</h1>
	</div>
	"""
	

	st.markdown(html_templ,unsafe_allow_html=True)
	st.write("Respiratory Disease Detection from Chest X-ray for Covid-19, Pneumonia, Lung Opacity or Healthy Cases")
	st.sidebar.image("CovidImage.jpg",width=300)
	image_file = st.sidebar.file_uploader("Upload an X-Ray Image (jpg, jpeg or png)",type=['jpg','jpeg','png'])

	if image_file is not None:
		our_image = Image.open(image_file)
		if st.sidebar.button("Image Preview"):
			st.sidebar.image(our_image,width=300)

		activities = ["Diagnosis", "Image Enhancement", "Disclaimer and Info"]
		choice = st.sidebar.selectbox("Select Activty",activities)

		if choice == 'Image Enhancement':
			st.subheader("Image Enhancement")

			enhance_type = st.sidebar.radio("Enhance Type",["Original","Contrast","Brightness"])
			if enhance_type == 'Contrast':

				c_rate = st.slider("Contrast",0.5,5.0)
				enhancer = ImageEnhance.Contrast(our_image)
				img_output = enhancer.enhance(c_rate)
				st.image(img_output,use_column_width=True)


			elif enhance_type == 'Brightness':

				c_rate = st.slider("Brightness",0.5,2.0)
				enhancer = ImageEnhance.Brightness(our_image)
				img_output = enhancer.enhance(c_rate)
				st.image(img_output,width=600,use_column_width=True)
				

			else:
				
				st.text("Original Image")
				st.image(our_image,width=600,use_column_width=True)



		elif choice == 'Diagnosis':
			pass

			if st.sidebar.button("Diagnosis"):
				#PreProcessing
				
				IMG_SIZE = (100,100)
				IMG_CHANNELS = 1
				
				new_img = np.array(our_image.convert('RGB'))
				new_img = cv2.cvtColor(new_img,IMG_CHANNELS)
				gray = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
				# st.text("Chest X-Ray")
				# st.image(gray,use_column_width=True)


				#X-Ray (Image) Preprocessing
 				
				img = cv2.equalizeHist(gray)
				img = cv2.resize(img,IMG_SIZE)
				X_Ray = img.reshape(-1,100,100,1)



				#Classification

				classes = ['covid', 'lung_opacity', 'normal', 'pneumonia']
				model = load_model("models/main_test_model_1/CNN_final_model.h5")


				#Prediction

				model_out = model.predict(X_Ray)[0]
				str_label = classes[np.argmax(model_out)]

				#Loading 
				my_bar = st.sidebar.progress(0)
				for percent_complete in range(100):
					time.sleep(0.005)
					my_bar.progress(percent_complete + 1)


				st.warning("This Web App is just a DEMO about Convolutional Neural Networks used to detect if the patient has respiratory Disease, The author is not a Doctor or Medical Professional !")


				if str_label == 'covid':

					
    				
					st.error("Diagnosis Result: Covid-19 is Detected in the chest X-ray of the patient")
					st.info("Top 5 Most Impacted Countries Live Covid-19 Status")
					st.write(df.head())
					st.subheader("Covid 19 Self Care Instructions")
					st.write("Self care for Asymptomatic cases, mild cases of COVID-19:")
					st.write("- Isolate yourself in a well ventilated room.")
					st.write("- Use a triple layer medical mask, discard mask after 8 hours of use or earlier if they become wet or visibly soiled. In the event of a caregiver entering the room, both caregiver and patient may consider using N 95 mask.")
					st.write("- Mask should be discarded only after disinfecting it with 1% Sodium Hypochlorite.")
					st.write("- Take rest and drink a lot of fluids to maintain adequate hydration.")
					st.write("- Follow respiratory etiquettes at all times.")
					st.write("- Frequent hand washing with soap and water for at least 40 seconds or clean with alcohol-based sanitizer.")
					st.write("- Don’t share personal items with other people in the household.")
					st.write("- Ensure cleaning of surfaces in the room that are touched often (tabletops, doorknobs, handles, etc.) with 1% hypochlorite solution.")
					st.write("- Monitor temperature daily.")
					st.write("- Monitor oxygen saturation with a pulse oximeter daily.")
					st.write("- Connect with the treating physician promptly if any deterioration of symptoms is noticed.")
					st.write("Instructions for caregivers:")
					st.write("- Mask: The caregiver should wear a triple layer medical mask. N95 mask may be considered when in the same room with the ill person.")
					st.write("- Hand hygiene: Hand hygiene must be ensured following contact with ill person or patient’s immediate environment.")
					st.write("- Exposure to patient/patient’s environment: Avoid direct contact with body fluids of the patient, particularly oral or respiratory secretions. Use disposable gloves while handling the patient. Perform hand hygiene before and after removing gloves.")
					


				elif str_label == 'lung_opacity':
					st.error("Diagnosis Result: Lung Opacity is Detected in the chest X-ray of the patient")
					st.subheader("Lung Opacity Self Care Instructions")
					st.write("- Always visit the doctor to get proper understanding of your illness and prescription for all of your medicines")
					st.write("- Drink plenty of fluids to help loosen secretions and bring up phlegm.")
					st.write("- Do not take cough medicines without first talking to your doctor. Coughing is one way your body works to get rid of an infection. If your cough is preventing you from getting the rest you need, ask your doctor about steps you can take to get relief.")
					st.write("- Drink warm beverages, take steamy baths and use a humidifier to help open your airways and ease your breathing. Contact your doctor right away if your breathing gets worse instead of better over time.")
					st.write("- Stay away from smoke to let your lungs heal. This includes smoking, secondhand smoke and wood smoke. Talk to your doctor if you are a smoker and are having trouble staying smokefree while you recover.")
					st.write("- Get lots of rest. You may need to stay in bed for a while. Get as much help as you can with meal preparation and household chores until you are feeling stronger. It is important not to overdo daily activities until you are fully recovered.")
				



				elif str_label == 'pneumonia':
					st.error("Diagnosis Result: Pneumonia is Detected in the chest X-ray of the patient")
					st.subheader("Pneumonia Self Care Instructions")
					
					
					st.write("- Always visit the doctor to get proper understanding of your illness and prescription for all of your medicines")
					st.write("- Control your fever with (with Professional Advise) aspirin, nonsteroidal anti-inflammatory drugs (NSAIDs, such as ibuprofen or naproxen), or acetaminophen. DO NOT give aspirin to children.")	
					st.write("- Drink plenty of fluids to help loosen secretions and bring up phlegm.")
					st.write("- Do not take cough medicines without first talking to your doctor. Coughing is one way your body works to get rid of an infection. If your cough is preventing you from getting the rest you need, ask your doctor about steps you can take to get relief.")
					st.write("- Drink warm beverages, take steamy baths and use a humidifier to help open your airways and ease your breathing. Contact your doctor right away if your breathing gets worse instead of better over time.")
					st.write("- Stay away from smoke to let your lungs heal. This includes smoking, secondhand smoke and wood smoke. Talk to your doctor if you are a smoker and are having trouble staying smokefree while you recover.")
					st.write("- Get lots of rest. You may need to stay in bed for a while. Get as much help as you can with meal preparation and household chores until you are feeling stronger. It is important not to overdo daily activities until you are fully recovered.")
					st.write("- If your pneumonia is so severe that you are treated in the hospital, you may be given intravenous fluids and antibiotics, as well as oxygen therapy, and possibly other breathing treatments.")
					


				else:
					st.success("Diagnosis Result: No disease Detected in the chest X-ray of the patient, Patient is Healthy")
					st.write("Self Care Instructions for Healthy Lungs")
					st.write("- STOP SMOKING: Smoking damages your lungs and increases your risk for a number of diseases including lung cancer and COPD. This is because combustion of materials releases harmful substances into your lungs (toxins and carcinogens). If you have never smoked, don't start. If you are still smoking, it’s never too late to quit. Learn more about how to quit including the many effective medications and smoking cessation programs that work.")
					st.write("- WASH YOUR HANDS: Wash thoroughly with soap and water several times a day to keep germs at bay and avoid most of the common infectious diseases that are spread by hand.")
					st.write("- CLEAN HOUSE: Air fresheners, mould, pet dander, and construction materials all pose a potential problem. Turn on the exhaust fan when you cook and avoid using aerosol products like hair spray.")
					st.write("- WEAR A MASK: Masks are a simple barrier to help prevent your respiratory droplets from reaching others. Studies show that masks reduce the spray of droplets when worn over the nose and mouth. You should wear a mask, even if you do not feel sick.")
					st.write("- TALK TO YOUR DOCTOR: See your doctor if you experience shortness of breath, pain when breathing, dizziness with a change of activity, a persistent cough, wheezing or coughing with exercise, pain in the airway.")
					st.write("- GET VACCINATED: This is especially important if you have lung disease, though healthy people also benefit from getting vaccinated. If you have significant lung disease or are over 65, a vaccination shot is highly recommended.")
					st.write("- STAYING HYDRATED: Staying well hydrated by taking in fluids throughout the day helps keep the mucosal linings in the lungs thin, This thinner lining helps the lungs function better.")
					st.write("- STAYING ACTIVE: Regular moderately intense activity is great for the lungs, Aim for at least least 20 minutes of consistent, moderately intense movement daily, like a brisk walk or cycle ride.")
					


				
				



		else:
			st.subheader("Disclaimer and Info")
			st.subheader("Disclaimer")
			st.write("**This App is used for demonstration of Deep Learning methodology known as Convolutional Neural Network on patients X-Ray for detection of several respiratory diseases, The author is not a Doctor and therefore there is no official medical recommendation provided by them**")
			st.write("**Please don't take the diagnosis outcome seriously the author is not a profession medical advisor therefore the results should not be considered valid!!!**")
			st.subheader("Info")
			st.write("The Application is created by Sumedh Vilas Kapse under the guidance of Prof Girish P Bhole")
			st.write("This App gets inspiration from the following works:")
			st.write("- [An Uncertainty-Aware Transfer Learning-Based Framework for COVID-19 Diagnosis](https://ieeexplore.ieee.org/document/9353390)") 
			st.write("- [Atomatic Detection and Diagnosis of Severe Viral Pneumonia CT Images Based on LDA-SVM](https://ieeexplore.ieee.org/document/8932625)") 
			st.write("- [sentdex Machine Learning/Deep Learning Information](https://www.youtube.com/user/sentdex)")
			st.write("We have used over 5000 X-Ray [images](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) of patients infected by Covid-19, Pneumonia, Lung Opacity and Healthy patients Collected from Kaggle which were developed by The Italian Society of Medical and Interventional Radiology (SIRM) COVID-19 DATABASE, Novel Corona Virus 2019 Dataset developed by Joseph Paul Cohen and Paul Morrison, and Lan Dao in GitHub")
			st.write("Dataset collected was of ample size to be used for classification problem. The result was quite good since we got 92.3% accuracy on the training set and 91% accuracy on the test set.")
			st.write("There are some drawbacks of the Application which would be improved upon further")
			st.write("- Unfortunately in our test set consisting of 1191 data, we got 45 cases of 'False Negative' for Covid-19 Prediction, 22 cases of 'False Negative' for Lung Opacity prediction, 24 cases of 'False Negative' for Pneumonia prediction and 12 cases of 'False Negative' for Healthy Patient Prediction. It's very easy to understand that these cases can be a huge issue.")
			st.write("- The Application could be expanded in the future for multiple diseases and the types of those diseases")
			st.write("- Collaboration with medical experts is needed to obtain better results for the application")






	if st.sidebar.button("About the Author"):
		st.sidebar.subheader("Respiratory Disease Detection Tool")
		st.sidebar.markdown("by [Sumedh Kapse](https://www.linkedin.com/in/sumedh-kapse-a0120a126/)")
		st.sidebar.markdown("[sumedhkapse9@gmail.com](mailto:sumedhkapse9@gmail.com)")
		

if __name__ == '__main__':
		main()	