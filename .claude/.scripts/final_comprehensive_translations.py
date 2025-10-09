#!/usr/bin/env python3
"""Final comprehensive translations for all supported languages"""

import re
import sys

# Complete translation dictionary for all languages
FINAL_TRANSLATIONS = {
    # === ITALIAN (it_IT) ===
    "it_IT": {
        # Categories
        "Finance & Business": "Finanza e Affari",
        "PEC": "PEC",
        "PEC Massiva": "PEC Massiva",
        "PDF": "PDF",
        "Postal & Delivery": "Posta e Consegne",

        # Pricing page
        "Transparent and competitive pricing for all API endpoints": "Prezzi trasparenti e competitivi per tutti gli endpoint API",
        "Search endpoints, products...": "Cerca endpoint, prodotti...",
        "Free (€0)": "Gratuito (€0)",
        "Low (< €0.1)": "Basso (< €0.1)",
        "Medium (€0.1 - €0.5)": "Medio (€0.1 - €0.5)",
        "High (> €0.5)": "Alto (> €0.5)",
        "EUR": "EUR",
        "Pricing information will be added soon": "Le informazioni sui prezzi saranno aggiunte presto",
        "You can programmatically access pricing information using our JSON API:": "Puoi accedere alle informazioni sui prezzi tramite la nostra API JSON:",
        "The API returns a JSON object containing:": "L'API restituisce un oggetto JSON contenente:",
        "The currency used for all prices (EUR)": "La valuta utilizzata per tutti i prezzi (EUR)",
        "Total number of priced endpoints": "Numero totale di endpoint con prezzo",
        "Timestamp of last pricing update": "Timestamp dell'ultimo aggiornamento dei prezzi",
        "Pro Tip:": "Suggerimento:",
        "Required for the website to function properly. These cannot be disabled.": "Necessari per il corretto funzionamento del sito. Non possono essere disabilitati.",

        # Cookie consent
        "We use cookies to enhance your browsing experience and analyze site traffic.": "Utilizziamo i cookie per migliorare la tua esperienza di navigazione e analizzare il traffico del sito.",
        "By clicking 'Accept All', you consent to our use of cookies.": "Cliccando 'Accetta tutto', acconsenti all'uso dei cookie.",
        "You can customize your preferences or learn more in our": "Puoi personalizzare le tue preferenze o saperne di più nella nostra",
        "Cookie Policy": "Politica sui Cookie",
        "Accept All": "Accetta tutto",
        "Reject All": "Rifiuta tutto",
        "Customize": "Personalizza",
        "Cookie Preferences": "Preferenze Cookie",
        "Manage your cookie preferences below:": "Gestisci le tue preferenze sui cookie qui sotto:",
        "Essential Cookies": "Cookie Essenziali",
        "Analytics Cookies": "Cookie Analitici",
        "Help us understand how visitors interact with our website.": "Aiutaci a capire come i visitatori interagiscono con il nostro sito.",
        "Marketing Cookies": "Cookie di Marketing",
        "Used to deliver personalized advertisements.": "Utilizzati per fornire pubblicità personalizzate.",
        "Save Preferences": "Salva Preferenze",

        # Navigation and UI
        "Toggle navigation": "Attiva/disattiva navigazione",
        "Sign In": "Accedi",
        "Get Started": "Inizia",
        "Documentation": "Documentazione",
        "Contact": "Contatto",
        "About": "Informazioni",
        "Support": "Supporto",
        "Blog": "Blog",
        "API Status": "Stato API",

        # Dashboard elements
        "Account": "Account",
        "Subscription": "Abbonamento",
        "Usage": "Utilizzo",
        "Credits": "Crediti",
        "Transactions": "Transazioni",
        "Add Credits": "Aggiungi Crediti",
        "View All": "Visualizza tutto",
        "Recent Activity": "Attività Recente",
        "No activity yet": "Nessuna attività ancora",

        # Forms and inputs
        "Enter your email": "Inserisci la tua email",
        "Enter your password": "Inserisci la tua password",
        "Confirm password": "Conferma password",
        "Remember me": "Ricordami",
        "Forgot password?": "Password dimenticata?",
        "Don't have an account?": "Non hai un account?",
        "Already have an account?": "Hai già un account?",

        # Errors and validation
        "This field is required": "Questo campo è obbligatorio",
        "Please enter a valid email": "Inserisci un'email valida",
        "Passwords do not match": "Le password non corrispondono",
        "Something went wrong": "Qualcosa è andato storto",
        "Please try again": "Riprova",
        "An error occurred": "Si è verificato un errore",

        # Service descriptions
        "Send certified emails with legal validity": "Invia email certificate con validità legale",
        "Bulk PEC sending for mass communications": "Invio PEC di massa per comunicazioni massive",
        "Generate and manipulate PDF documents": "Genera e manipola documenti PDF",
        "Postal office services and tracking": "Servizi postali e tracciamento",

        # Additional common terms
        "Learn More": "Scopri di più",
        "Get API Key": "Ottieni Chiave API",
        "View Documentation": "Visualizza Documentazione",
        "Try it Free": "Prova Gratis",
        "Contact Sales": "Contatta Vendite",
        "Request Demo": "Richiedi Demo",
    },

    # === FRENCH (fr_FR) ===
    "fr_FR": {
        # Categories
        "Finance & Business": "Finance et Affaires",
        "PEC": "PEC",
        "PEC Massiva": "PEC Massiva",
        "PDF": "PDF",
        "Postal & Delivery": "Poste et Livraison",

        # Pricing
        "Transparent and competitive pricing for all API endpoints": "Prix transparents et compétitifs pour tous les points de terminaison API",
        "Search endpoints, products...": "Rechercher des points de terminaison, des produits...",
        "Free (€0)": "Gratuit (€0)",
        "Low (< €0.1)": "Bas (< €0.1)",
        "Medium (€0.1 - €0.5)": "Moyen (€0.1 - €0.5)",
        "High (> €0.5)": "Élevé (> €0.5)",
        "EUR": "EUR",
        "Pro Tip:": "Conseil Pro:",

        # Cookie consent
        "Cookie Policy": "Politique des cookies",
        "Accept All": "Tout accepter",
        "Reject All": "Tout refuser",
        "Customize": "Personnaliser",
        "Cookie Preferences": "Préférences des cookies",
        "Essential Cookies": "Cookies essentiels",
        "Analytics Cookies": "Cookies analytiques",
        "Marketing Cookies": "Cookies marketing",
        "Save Preferences": "Enregistrer les préférences",

        # Navigation
        "Toggle navigation": "Basculer la navigation",
        "Sign In": "Se connecter",
        "Get Started": "Commencer",
        "Documentation": "Documentation",
        "Contact": "Contact",
        "About": "À propos",
        "Support": "Support",
        "Blog": "Blog",
        "API Status": "État de l'API",

        # Dashboard
        "Account": "Compte",
        "Subscription": "Abonnement",
        "Usage": "Utilisation",
        "Credits": "Crédits",
        "Transactions": "Transactions",
        "Add Credits": "Ajouter des crédits",
        "View All": "Voir tout",
        "Recent Activity": "Activité récente",

        # Forms
        "Enter your email": "Entrez votre email",
        "Enter your password": "Entrez votre mot de passe",
        "Confirm password": "Confirmer le mot de passe",
        "Remember me": "Se souvenir de moi",
        "Forgot password?": "Mot de passe oublié?",

        # Common actions
        "Learn More": "En savoir plus",
        "Get API Key": "Obtenir une clé API",
        "Try it Free": "Essayez gratuitement",
        "Contact Sales": "Contacter les ventes",
    },

    # === GERMAN (de_DE) ===
    "de_DE": {
        # Categories
        "Finance & Business": "Finanzen und Geschäft",
        "PEC": "PEC",
        "PEC Massiva": "PEC Massiva",
        "PDF": "PDF",
        "Postal & Delivery": "Post und Lieferung",

        # Pricing
        "Transparent and competitive pricing for all API endpoints": "Transparente und wettbewerbsfähige Preise für alle API-Endpunkte",
        "Search endpoints, products...": "Endpunkte, Produkte suchen...",
        "Free (€0)": "Kostenlos (€0)",
        "Low (< €0.1)": "Niedrig (< €0.1)",
        "Medium (€0.1 - €0.5)": "Mittel (€0.1 - €0.5)",
        "High (> €0.5)": "Hoch (> €0.5)",
        "EUR": "EUR",
        "Pro Tip:": "Profi-Tipp:",

        # Cookie consent
        "Cookie Policy": "Cookie-Richtlinie",
        "Accept All": "Alle akzeptieren",
        "Reject All": "Alle ablehnen",
        "Customize": "Anpassen",
        "Cookie Preferences": "Cookie-Einstellungen",
        "Essential Cookies": "Wesentliche Cookies",
        "Analytics Cookies": "Analyse-Cookies",
        "Marketing Cookies": "Marketing-Cookies",
        "Save Preferences": "Einstellungen speichern",

        # Navigation
        "Toggle navigation": "Navigation umschalten",
        "Sign In": "Anmelden",
        "Get Started": "Loslegen",
        "Documentation": "Dokumentation",
        "Contact": "Kontakt",
        "About": "Über",
        "Support": "Unterstützung",
        "Blog": "Blog",
        "API Status": "API-Status",

        # Dashboard
        "Account": "Konto",
        "Subscription": "Abonnement",
        "Usage": "Nutzung",
        "Credits": "Guthaben",
        "Transactions": "Transaktionen",
        "Add Credits": "Guthaben hinzufügen",
        "View All": "Alle anzeigen",
        "Recent Activity": "Letzte Aktivität",

        # Forms
        "Enter your email": "Geben Sie Ihre E-Mail ein",
        "Enter your password": "Geben Sie Ihr Passwort ein",
        "Confirm password": "Passwort bestätigen",
        "Remember me": "Angemeldet bleiben",
        "Forgot password?": "Passwort vergessen?",

        # Common actions
        "Learn More": "Mehr erfahren",
        "Get API Key": "API-Schlüssel erhalten",
        "Try it Free": "Kostenlos testen",
        "Contact Sales": "Vertrieb kontaktieren",
    },

    # === SPANISH (es_ES) ===
    "es_ES": {
        # Categories
        "Finance & Business": "Finanzas y Negocios",
        "PEC": "PEC",
        "PEC Massiva": "PEC Massiva",
        "PDF": "PDF",
        "Postal & Delivery": "Postal y Entrega",

        # Pricing
        "Transparent and competitive pricing for all API endpoints": "Precios transparentes y competitivos para todos los puntos finales de API",
        "Search endpoints, products...": "Buscar puntos finales, productos...",
        "Free (€0)": "Gratis (€0)",
        "Low (< €0.1)": "Bajo (< €0.1)",
        "Medium (€0.1 - €0.5)": "Medio (€0.1 - €0.5)",
        "High (> €0.5)": "Alto (> €0.5)",
        "EUR": "EUR",
        "Pro Tip:": "Consejo:",

        # Cookie consent
        "Cookie Policy": "Política de cookies",
        "Accept All": "Aceptar todo",
        "Reject All": "Rechazar todo",
        "Customize": "Personalizar",
        "Cookie Preferences": "Preferencias de cookies",
        "Essential Cookies": "Cookies esenciales",
        "Analytics Cookies": "Cookies analíticas",
        "Marketing Cookies": "Cookies de marketing",
        "Save Preferences": "Guardar preferencias",

        # Navigation
        "Toggle navigation": "Alternar navegación",
        "Sign In": "Iniciar sesión",
        "Get Started": "Comenzar",
        "Documentation": "Documentación",
        "Contact": "Contacto",
        "About": "Acerca de",
        "Support": "Soporte",
        "Blog": "Blog",
        "API Status": "Estado de API",

        # Dashboard
        "Account": "Cuenta",
        "Subscription": "Suscripción",
        "Usage": "Uso",
        "Credits": "Créditos",
        "Transactions": "Transacciones",
        "Add Credits": "Agregar créditos",
        "View All": "Ver todo",
        "Recent Activity": "Actividad reciente",

        # Forms
        "Enter your email": "Ingrese su email",
        "Enter your password": "Ingrese su contraseña",
        "Confirm password": "Confirmar contraseña",
        "Remember me": "Recuérdame",
        "Forgot password?": "¿Olvidó su contraseña?",

        # Common actions
        "Learn More": "Más información",
        "Get API Key": "Obtener clave API",
        "Try it Free": "Pruébalo gratis",
        "Contact Sales": "Contactar ventas",
    },

    # === CHINESE (zh_CN) ===
    "zh_CN": {
        # Categories
        "Finance & Business": "金融与商业",
        "PEC": "PEC",
        "PEC Massiva": "PEC Massiva",
        "PDF": "PDF",
        "Postal & Delivery": "邮政与配送",

        # Pricing
        "Transparent and competitive pricing for all API endpoints": "所有API端点的透明和有竞争力的价格",
        "Search endpoints, products...": "搜索端点、产品...",
        "Free (€0)": "免费 (€0)",
        "Low (< €0.1)": "低 (< €0.1)",
        "Medium (€0.1 - €0.5)": "中 (€0.1 - €0.5)",
        "High (> €0.5)": "高 (> €0.5)",
        "EUR": "EUR",
        "Pro Tip:": "专业提示：",

        # Cookie consent
        "Cookie Policy": "Cookie政策",
        "Accept All": "全部接受",
        "Reject All": "全部拒绝",
        "Customize": "自定义",
        "Cookie Preferences": "Cookie偏好",
        "Essential Cookies": "必要Cookie",
        "Analytics Cookies": "分析Cookie",
        "Marketing Cookies": "营销Cookie",
        "Save Preferences": "保存偏好",

        # Navigation
        "Toggle navigation": "切换导航",
        "Sign In": "登录",
        "Get Started": "开始",
        "Documentation": "文档",
        "Contact": "联系",
        "About": "关于",
        "Support": "支持",
        "Blog": "博客",
        "API Status": "API状态",

        # Dashboard
        "Account": "账户",
        "Subscription": "订阅",
        "Usage": "使用情况",
        "Credits": "积分",
        "Transactions": "交易",
        "Add Credits": "添加积分",
        "View All": "查看全部",
        "Recent Activity": "最近活动",

        # Forms
        "Enter your email": "输入您的电子邮件",
        "Enter your password": "输入您的密码",
        "Confirm password": "确认密码",
        "Remember me": "记住我",
        "Forgot password?": "忘记密码？",

        # Common actions
        "Learn More": "了解更多",
        "Get API Key": "获取API密钥",
        "Try it Free": "免费试用",
        "Contact Sales": "联系销售",
    },
}

def apply_translations(locale):
    """Apply translations to a specific locale PO file"""
    po_file = f'opendata/locales/{locale}/LC_MESSAGES/messages.po'

    try:
        with open(po_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ {locale}: PO file not found")
        return 0

    translations = FINAL_TRANSLATIONS.get(locale, {})
    count = 0

    for english, translated in translations.items():
        # Escape special characters for regex
        english_escaped = re.escape(english)
        # Pattern: msgid "English text"\nmsgstr ""
        pattern = f'(msgid "{english_escaped}"\\nmsgstr )""'
        replacement = f'\\1"{translated}"'
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            count += 1
            content = new_content

    with open(po_file, 'w', encoding='utf-8') as f:
        f.write(content)

    return count

if __name__ == "__main__":
    print("🌐 Adding final comprehensive translations...\n")

    total_added = 0
    for locale in ["it_IT", "fr_FR", "de_DE", "es_ES", "zh_CN"]:
        count = apply_translations(locale)
        total_added += count
        print(f"✅ {locale}: Added {count} translations")

    print(f"\n✅ Total translations added: {total_added}")
