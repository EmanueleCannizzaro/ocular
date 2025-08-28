
from pydantic import BaseModel, Field, create_model
from typing import Optional, Dict, Any, List


class DocumentFileChunk(BaseModel):
    base64: str
    mime_type: str

# 1. Define the deepest nested models
class DettagliTrascrizione(BaseModel):
    data: str
    registro_particolare: str
    registro_generale: str

class AttoPignoramento(BaseModel):
    data: str
    numero_repertorio: str
    dettagli_trascrizione: DettagliTrascrizione

class ProceduraGiudiziaria(BaseModel):
    tribunale: str
    atto_pignoramento: AttoPignoramento

class StatoConservazione(BaseModel):
    edificio_condominiale: str
    appartamento: str

class DescrizioneImmobile(BaseModel):
    tipo_proprieta: str
    indirizzo: str
    superficie_commerciale_mq: float
    composizione_interna: str
    stato_conservazione: StatoConservazione

class DatiCatastali(BaseModel):
    catasto: str
    sezione: str
    foglio: str
    particella: str
    subalterno: str
    zona_censuaria: str
    categoria: str
    classe: str
    vani: int
    rendita_catastale_euro: float

class DetrazioniEuro(BaseModel):
    sanatoria_catastale_urbanistica: float
    riparazioni_strutturali_interne: float

class ValoreMercatoStimato(BaseModel):
    valore_lordo_euro: float
    detrazioni_euro: DetrazioniEuro
    valore_netto_euro: float
    note_valutazione: str

class ValutazioneEconomica(BaseModel):
    valore_di_mercato_stimato: ValoreMercatoStimato
    prezzo_base_asta_euro: float

class StatoOccupazione(BaseModel):
    stato: str
    implicazioni: str

class CostiOcculti(BaseModel):
    costo_sanatoria_riparazioni_euro: float
    debito_condominiale_pregresso_euro: float
    nota_condominio: str

class RischiLegali(BaseModel):
    difformita_riscontrate: List[str]
    controversie: str

class OneriERischi(BaseModel):
    stato_occupazione: StatoOccupazione
    costi_occulti: CostiOcculti
    rischi_legali: RischiLegali
    ipoteche_iscrizioni: List[str]

class PotenzialeConvenienza(BaseModel):
    valore_risparmio_euro: float
    percentuale_risparmio: str

class AnalisiInvestimento(BaseModel):
    potenziale_convenienza: PotenzialeConvenienza
    totale_costi_aggiuntivi_stimati: float
    profilo_rischio: str
    considerazioni_finali: str

# 2. Define the main top-level model
class InvestimentoImmobiliare(BaseModel):
    riassunto_perizia: str
    identificativo: str
    procedura_giudiziaria: ProceduraGiudiziaria
    descrizione_immobile: DescrizioneImmobile
    dati_catastali: DatiCatastali
    valutazione_economica: ValutazioneEconomica
    oneri_e_rischi: OneriERischi
    analisi_investimento: AnalisiInvestimento

# 3. Define the root model for the entire JSON
class RootModel(BaseModel):
    investimento_immobiliare: InvestimentoImmobiliare
