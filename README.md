# :computer: PMU-B-PersonalAI (E-SAN Thailand Coding & AI Academy ) :computer:

## Introduction
### ชื่อ-สกุล : ชิติพัทธ์ สร้อยสังวาลย์
### ระดับการศึกษา : ชั้นมัธยมศึกษาปีที่ 5 โรงเรียนสตรีวิทยา 2
### ความสนใจ : Large Language Model (LLM) 


[Video แนะนำตัว](https://youtu.be/-XFoQNt75qQ)
--

## Chapter I : xPore: An AI-Powered App for Bioinformaticians
โดยเนื้อหาจะเกี่ยวกับการแปลง mRNA เป็น Gene โดยผ่าน nanopore sequence แล้วจะสามารถอธิบายยีนเป็นโรคอะไรหรือมีฟังก์ชันอะไร
  
Exercise and Workshop : [Gaussian Mixture Model](https://github.com/KamiK4M1/PMU-B-PersonalAI/blob/main/xPore%20An%20AI-Powered%20App/GMM.ipynb)

จากการทำ Exercise พบว่ากราฟที่ยังใช้การแจกแจงปกติเปลี่ยนไปเนื่องจากค่าเฉลี่ย (Mean) และส่วนเบี่ยงเบนมาตรฐาน (S.D.) เปลี่ยนตามโมเดลของ Gaussian Mixture

---

## Chapter II: Learning from Biosignal
โดยเนื้อหาจะเกี่ยวกับการวิเคราะห์ระยะการนอนทั้ง 5 โดยใช้คลื่นสมองว่า ณ ตอนนี้เป็นระยะการนอนใด


  
Exercise and Workshop : [1D CNN for brain signal](https://github.com/KamiK4M1/PMU-B-PersonalAI/tree/main/Learning%20from%20Biosignal/Exercise-pmub-learning-biosignals)

จากการทำ Exercise ได้ลองทำบน Vscode และแก้ไข model.py ตามรูปแบบ tinysleepnet เรียบร้อย

---

## Chapter III: AI for detecting code plagiarism 
โดยเนื้อหาจะเกี่ยวกับการตรวจสอบว่าโค้ดที่นับว่าเปรียบเทียบมีการคัดลอก (Plagiarism) หรือไม่

  
Exercise and Workshop : [Code2Vec to detect code clone](https://github.com/KamiK4M1/PMU-B-PersonalAI/blob/main/AI%20for%20detecting%20code/PMU_B_CodingAI_CodeCloneDetection_Workshop.ipynb)

จากการทำ Exercise ได้ค่า Cosine ที่เปรียบเทียบกันทั้ง 16 คู่ด้วยวิธีอย่างง่ายคือการนำโค้ดมาเปลี่ยนค่าของแต่ละคู่ ทั้งนี้วิธีการดังกล่าวมีทั้งข้อดีและข้อเสีย

---

## Chapter IV: BiTNet: AI for diagnosing ultrasound image 
โดยเนื้อหาจะเกี่ยวกับการตรวจบริเวณช่องท้องว่ามีสิ่งผิดปกติหรือไม่ ด้วยการใช้ ultrasound ที่ใช้มุมแตกต่างกันออกไป

 Exercise and Workshop : [EffcientNet: Image Classificaiton](https://github.com/KamiK4M1/PMU-B-PersonalAI/blob/main/BiTNet%20AI%20for%20diagnosing/PMUB_Personal_AI_Image_classification_EfficientNetB5.ipynb)

จากการทำ Exercise พบว่ารูปภาพสุนัขและแมวจะทำนายได้ดี ในขณะที่ภาพนก เครื่องบินและอื่นๆก็ยังทำนายเป็นสุนัขและแมวเช่นกันเนื่องจากเราสอนให้รู้จักแต่สุนัขและแมว และได้พบว่าค่าใดคือค่า Output ของการทำโมเดล

---

## Chapter V: Mental disorder detection from social media data

โดยเนื้อหาจะเกี่ยวกับการตรวจสอบข้อมูลและทำนายความรู้สึกจากข้อมูล Social Media ว่า ณ ตอนนี้ผู้ใช้รายนั้นรู้สึกอย่างไร
  
  
Exercise and Workshop : [NLP classification](https://github.com/KamiK4M1/PMU-B-PersonalAI/blob/main/Mental%20disorder%20detection/E_san_coding.ipynb)

จากการทำ Exercise ได้ทำการเลือกใช้โมเดลอื่นๆเช่น SVC และ KNNRegression ในการสร้างโมเดลขึ้นมาเพิ่มแต่ และได้ลองเก็บ Report พบว่าโมเดล Logistic Regression สามารถทำนายได้่ดีกว่า 2 ตัวที่เลือกมา

---

## Chapter VI: AI for arresting criminals
โดยเนื้อหาจะเกี่ยวกับการตรวจสอบบุคคลและรถที่เป็นผู้ต้องสงสัยที่อาจก่อเหตุอาชญากรรมด้วยวิดิโอและรูปภาพของกล้องวงจรปิด
  
Exercise and Workshop : [Yolo Detection](https://github.com/KamiK4M1/PMU-B-PersonalAI/blob/main/AI%20for%20arresting%20criminals/Train_Yolov8_Object_Detection_on_Custom_Dataset.ipynb)

จากการทำ Exercise ได้พบว่ายังมี Taxi หลายคันถูกนับว่าเป็น car และความน่าเชื่อถือของ pedestain มีน้อยเนื่องจากข้อมูลที่ Detection ให้น้อยเกินไปจนไม่สามารถตัดสินได้หรือตัดสินใจได้น้อย
