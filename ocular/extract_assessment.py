import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

from ocular.models.property_assessment import (
    AppraisalSummary, 
    AuctionDetails, AuctionPropertyProfile, 
    CadastralData, 
    Person,  
    PropertyAsset, PublicationWebsite, 
    RiskAssessment, ValuationDetails, 
)


# --- 2. CONFIGURAZIONE DEL CLIENT E PATCH CON INSTRUCTOR ---

# Configura il client per puntare al tuo server Ollama locale
client = instructor.patch(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # 'ollama' è una chiave fittizia richiesta da Ollama
    ),
    mode=instructor.Mode.JSON,
)


# --- 3. DATI DI INPUT (TESTO ESTRATTO DAL PDF) ---

# Ho copiato qui il testo completo della perizia che hai fornito.
# In un'applicazione reale, questo testo verrebbe letto da un file.
document_text = """
TRIBUNALE DI SPOLETO
Giudice: Dott. Alberto CAPPELLINI
RELAZIONE
Esecuzione Immobiliare
R.G. 102/2022
... [INSERIRE QUI IL TESTO COMPLETO DELLA PERIZIA] ...
Il valore totale dei manufatti quantificato in totale di € 299.775,00 viene adeguato...
Pertanto il valore a base d'asta sarà: 269.797,50 €
L' Esperto
Dott. Agr. Carlo BONIFAZI MEFFE
"""


# --- 4. ESECUZIONE DELL'ESTRAZIONE ---

def extract_appraisal_data(content: str) -> AuctionPropertyProfile:
    """
    Funzione per estrarre i dati da un testo di perizia usando il modello LLM
    e forzando l'output nel formato Pydantic AuctionPropertyProfile.
    """
    print("Avvio dell'estrazione dei dati dalla perizia...")
    try:
        profile = client.chat.completions.create(
            model="llama3:8b",
            response_model=AuctionPropertyProfile,
            messages=[
                {
                    "role": "system",
                    "content": "Sei un esperto analista immobiliare specializzato in aste giudiziarie italiane. Il tuo compito è estrarre in modo accurato e completo le informazioni da una perizia tecnica e strutturarle secondo lo schema JSON fornito. Presta la massima attenzione a tutti i dettagli, inclusi dati catastali, vincoli legali, costi di sanatoria e calcoli di valutazione.",
                },
                {
                    "role": "user",
                    "content": f"Analizza la seguente perizia tecnica e popola tutti i campi del modello di dati:\n\n---\n\n{content}"
                },
            ],
            max_retries=2, # Tenta nuovamente se la validazione Pydantic fallisce
        )
        print("Estrazione completata con successo!")
        return profile
    except Exception as e:
        print(f"Errore durante l'estrazione: {e}")
        return None

if __name__ == "__main__":
    # Per eseguire questo script, sostituisci il testo troncato in `document_text`
    # con il contenuto completo della perizia.
    if "... [INSERIRE QUI IL TESTO COMPLETO DELLA PERIZIA] ..." in document_text:
        print("ATTENZIONE: Inserire il testo completo della perizia nella variabile `document_text` per eseguire l'estrazione.")
    else:
        structured_data = extract_appraisal_data(document_text)
        if structured_data:
            # Stampa il JSON strutturato e ben formattato
            print("\n--- DATI ESTRATTI ---")
            print(structured_data.model_dump_json(indent=2))