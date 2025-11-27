
# 🏥 MediQueue

**AI-Driven and Secure Hospital Emergency Triage System**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Core%20Component-AI%20Triage-orange)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-teal)
![Status](https://img.shields.io/badge/Project-Senior%20Project-success)

MediQueue is an **AI-powered emergency department triage system** that uses **machine-learning models to automatically classify patient urgency** based on real clinical data.
The system is designed to **support medical staff**, reduce waiting time, and improve triage accuracy in high-pressure emergency environments.

> 🎓 **Senior Project (IS499)** — Prince Sultan University
> 🧠 **Primary Focus: AI-Based Patient Severity Classification**

---

## 🎯 Why AI in MediQueue?

Manual triage methods are:

* Time-consuming
* Subjective
* Prone to inconsistency during high workload periods

MediQueue introduces **machine-assisted intelligence** to ensure:

* Faster triage decisions
* More consistent severity categorization
* Reduced human error under pressure

The AI system **assists** healthcare professionals — it does **not replace clinical judgment**.

---

## 🧠 AI Triage Engine (Core System)

The AI component is the **heart of MediQueue**.

### ✅ AI Responsibilities

* Analyze patient vital signs and symptoms
* Automatically predict patient severity level
* Instantly return results to the triage dashboard
* Maintain consistency across triage decisions

### ✅ Severity Levels Predicted

* **Critical**
* **Emergency**
* **Urgent**
* **Semi-Urgent**
* **Non-Urgent**

These labels directly influence patient prioritization in the emergency queue.

---

## ⚙️ How the AI Model Works

### 1️⃣ Data Input

The model receives structured patient data including:

* Heart rate
* Blood pressure
* Body temperature
* Oxygen saturation
* Pain level
* Consciousness level
* Presenting symptoms

---

### 2️⃣ Data Preprocessing

Before prediction, the system:

* Normalizes numerical features
* Encodes categorical variables
* Aligns inputs with trained feature structure (`feature_columns.pkl`)
* Applies scaling (`scaler.pkl`)

---

### 3️⃣ Machine Learning Prediction

* A trained **classification model** processes the input
* Outputs a predicted severity level
* Uses pre-trained encoders and mappings

Prediction time is **near-instant**, ensuring system responsiveness during emergencies.

---

### 4️⃣ Decision Support Output

* Severity result is displayed immediately
* Sent to doctors’ and nurses’ dashboards
* Used to reorder patient queue

✅ The AI provides **decision support**, not autonomous decisions.

---


## 👥 System Users

| Role        | Interaction with AI                          |
| ----------- | -------------------------------------------- |
| **Patient** | Receives severity and wait status            |
| **Nurse**   | Inputs vitals → triggers AI triage           |
| **Doctor**  | Reviews AI output to support diagnosis       |
| **Admin**   | Monitors AI predictions and system analytics |

---

## 🛠 Tech Stack

**AI & Data Science**

* Python
* Scikit-learn
* Pandas
* NumPy

**Backend**

* FastAPI

**Frontend**

* HTML5 · CSS3 · JavaScript

**Security**

* Role-Based Access Control (RBAC)
* Password hashing

---

## ▶️ Run Locally

```bash
git clone https://github.com/GhalaAlkhorayef/MediQueue.git
cd MediQueue

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
uvicorn app:app --reload
```

---

## 👩‍💻 Team

* **Ghala Alkhorayef**
* Alanoud Alassaf
* Hissah Alsuhaibani
* Showq Alhussaini
* Areeb Bintashlan


**Supervisor:**
Dr. Nor Shahida Jamail
Prince Sultan University

---

## 📄 Academic Notice

This project was developed for **educational and research purposes only** and does not handle real patient records.

---

