from pydantic import BaseModel, Field
from typing import List, Literal, Optional

# --- MODELLI DI BASE ---

class Person(BaseModel):
    """Rappresenta una persona coinvolta nel processo d'asta."""
    full_name: str = Field(..., description="Il nome completo della persona.")
    role: Literal["Judge", "Delegate", "Custodian", "Expert Appraiser"] = Field(..., description="Il ruolo della persona nella procedura.")
    email: Optional[str] = Field(description="Indirizzo email di contatto, se disponibile.")
    phone: Optional[str] = Field(description="Numero di telefono di contatto, se disponibile.")

class CadastralData(BaseModel):
    """Rappresenta i dati catastali italiani."""
    sheet: str = Field(..., description="Il numero del foglio catastale ('Foglio').")
    section: Optional[str] = Field(description="La sezione catastale ('Sezione').")
    subsection: Optional[str] = Field(description="La sottosezione catastale ('Sottosezione').")
    parcel: str = Field(..., description="Il numero della particella catastale ('Particella').")
    subordinate: Optional[str] = Field(description="Il numero del subalterno, se applicabile.")
    category: Optional[str] = Field(description="La categoria catastale (es. 'A/3', 'C/6').")
    revenue: Optional[float] = Field(description="La rendita catastale in Euro ('Rendita').")

class PropertyAsset(BaseModel):
    """Rappresenta un singolo bene (fabbricato o terreno) all'interno del lotto d'asta."""
    asset_type: str = Field(..., description="La tipologia del bene (es. 'Abitazione', 'Garage', 'Terreno', 'Fabbricati Costruiti Per Esigenze Industriali').")
    description: str = Field(..., description="La descrizione dettagliata di questo specifico bene.")
    address: str = Field(..., description="L'indirizzo completo di questo specifico bene.")
    surface_sqm: float = Field(..., description="La superficie in mq per questo bene.")
    cadastral_data: Optional[CadastralData] = Field(description="I dati catastali specifici per questo bene, se disponibili.")

class PublicationWebsite(BaseModel):
    """Rappresenta un sito web dove l'annuncio d'asta è stato pubblicato."""
    site_name: str = Field(..., description="Il nome del sito web (es. 'www.asteavvisi.it').")
    listing_url: str = Field(..., description="L'URL diretto dell'annuncio d'asta su quel sito.")

# --- SEZIONI DETTAGLIATE DEL MODELLO ---

class AuctionDetails(BaseModel):
    """Contiene tutte le informazioni relative al processo d'asta dal portale vendite."""
    sale_type: str = Field(..., description="La descrizione del tipo di vendita ('descTipoVendita'), es. 'Senza Incanto'.")
    sale_mode: str = Field(..., description="La modalità di vendita ('descModVendita'), es. 'Sincrona Mista'.")
    publication_date: str = Field(..., description="La data di pubblicazione dell'annuncio ('dataDiPubblicazione').")
    offer_submission_deadline: str = Field(..., description="La data e l'ora di scadenza per la presentazione delle offerte ('dataTermPresOff' e 'oraTermPresOff').")
    auction_date: str = Field(..., description="La data e l'ora della vendita ('dataVendita' e 'oraVendita').")
    base_price: float = Field(..., description="Il prezzo base d'asta ('impoBaseAsta').")
    minimum_bid: float = Field(..., description="L'offerta minima accettabile ('impoOffertaMinima').")
    minimum_bid_increase: Optional[float] = Field(description="L'incremento minimo per il rilancio ('impoOffertaAumento').")
    required_deposit_rule: str = Field(..., description="La regola per il calcolo della cauzione ('depoCauzionale'), es. '20% prezzo offerto'.")
    
class AppraisalSummary(BaseModel):
    """Contiene i dettagli principali della perizia tecnica."""
    court: str = Field(..., description="Il tribunale di competenza (es. 'TRIBUNALE DI SPOLETO').")
    procedure_number: str = Field(..., description="Il numero di registro della procedura (es. 'R.G. 102/2022').")
    judge: Person = Field(..., description="Il Giudice dell'Esecuzione.")
    expert_appraiser: Person = Field(..., description="L'esperto (CTU) che ha redatto la perizia.")

class Lien(BaseModel):
    """Rappresenta un'iscrizione o trascrizione pregiudizievole come un'ipoteca o un pignoramento."""
    lien_type: str = Field(..., description="Il tipo di vincolo (es. 'Ipoteca Volontaria', 'Pignoramento').")
    registration_date: str = Field(..., description="La data di iscrizione o trascrizione.")
    beneficiary: Optional[str] = Field(description="Il soggetto a favore del quale il vincolo è stato iscritto.")
    amount: Optional[float] = Field(description="L'importo capitale del vincolo in Euro.")

class RemediationCost(BaseModel):
    """Rappresenta un costo stimato per sanare una non conformità."""
    description: str = Field(..., description="La descrizione dell'intervento necessario (es. 'Aggiornamento catastale').")
    estimated_cost: float = Field(..., description="Il costo stimato in Euro per la sanatoria.")

class RiskAssessment(BaseModel):
    """Una sezione dedicata all'analisi dei rischi specifici dell'immobile."""
    seismic_risk_notes: Optional[str] = Field(description="Note sul rischio sismico della zona e sugli interventi strutturali passati (es. ristrutturazione post-terremoto).")
    compliance_issues: List[str] = Field(..., description="Un elenco testuale dei problemi di conformità, come la mancanza del certificato di agibilità, le discrepanze catastali e le strutture abusive da rimuovere.")
    costs_to_remedy: List[RemediationCost] = Field(..., description="Un elenco strutturato dei costi necessari per regolarizzare l'immobile.")
    liens_and_encumbrances: List[Lien] = Field(..., description="Un elenco delle iscrizioni e trascrizioni pregiudizievoli (ipoteche, pignoramenti).")

class ValuationDetails(BaseModel):
    """Contiene il calcolo dettagliato del valore dell'immobile come da perizia."""
    total_market_value: float = Field(..., description="Il valore di mercato totale stimato dal perito.")
    adjustment_percentage: float = Field(..., description="La percentuale di abbattimento forfettario applicata (es. -10.0 per un 10%).")
    final_auction_base_price: float = Field(..., description="Il valore finale a base d'asta calcolato nella perizia.")

# --- MODELLO PRINCIPALE ---

class AuctionPropertyProfile(BaseModel):
    """
    Il modello Pydantic definitivo che combina i dati del portale (JSON) e della perizia (PDF).
    """
    sale_id: int = Field(..., description="L'ID univoco della vendita dal file JSON ('idVendita').")
    lot_number: str = Field(..., description="Il numero di lotto specifico ('codLotto').")
    
    appraisal_summary: AppraisalSummary
    auction_details: AuctionDetails
    risk_assessment: RiskAssessment
    valuation_details: ValuationDetails
    
    assets: List[PropertyAsset] = Field(..., description="Un elenco di tutti i beni che compongono il lotto.")
    occupancy_status: str = Field(..., description="Lo stato di occupazione dell'immobile (es. 'Libero', 'Occupato dal debitore').")
    publication_websites: List[PublicationWebsite] = Field(..., description="Un elenco dei siti web dove l'asta è pubblicata.")
