"""
AI Service for Easy Tables
Handles AI processing with custom prompt for data extraction and standardization
"""

import json
import logging
from typing import Dict, Any, Optional, List
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError

from ..core.config import settings
from ..core.utils import get_desired_headers, get_period_display_name


logger = logging.getLogger(__name__)


class AIService:
    """Service for AI processing using custom prompt for financial data extraction"""
    
    def __init__(self):
        self.client = None
        if settings.azure_ai_foundry_connection and settings.azure_ai_foundry_key:
            try:
                self.client = ChatCompletionsClient(
                    endpoint=settings.azure_ai_foundry_connection,
                    credential=AzureKeyCredential(settings.azure_ai_foundry_key)
                )
                logger.info("Azure AI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Azure AI client: {e}")
        else:
            logger.warning("Azure AI credentials not provided")
    
    def _get_custom_prompt(
        self, 
        file_type: str, 
        period: str, 
        is_iv: bool,
        extracted_text: str
    ) -> str:
        """Generate the custom prompt for AI processing"""
        
        period_display = get_period_display_name(period)
        desired_headers = get_desired_headers(file_type)
        
        prompt = f"""Tvoj zadatak je da iz ulaznog fajla izdvojiš podatke relevantne za analitičke kartice, standardizuješ ih i vratiš u JSON formatu.

Ulazne promenljive:

file_type: "{file_type}"
period_date: "{period_display}"
desired_headers: {desired_headers}
is_ispravka_vrednosti_kupci: {is_iv}

Pravila obrade:

Tabela

Izdvoji iz fajla samo tabele koje sadrže finansijske podatke vezane za file_type.

Ignoriši sve druge tabele (npr. opisi, reference, šifre konta koje nisu u obliku tabele prometa).

Standardizuj kolone prema desired_headers.

Konto

Ako ulazna tabela nema kolonu konto, sam je dopuni na osnovu dostupnih informacija (ako postoji u tekstu ili zaglavljima).

Saldo i promet

Uporebi samo relevantne kolone u skladu sa desired_headers (ako je kupac → "promet duguje", ako je dobavljač → "promet potražuje").

saldo je obavezno polje.

Datum

Izdvoji datum iz fajla. Datum se može nalaziti bilo gde (najčešće na početku dokumenta).

Vrati ga u formatu YYYY-MM-DD.

Polje u JSON-u neka bude "datum": "...", nezavisno od toga da li je period_date kraj fiskalne godine ili presek bilansa (taj info ti već znaš iz varijable period_date).

Ispravka vrednosti kupaca

Ako je is_ispravka_vrednosti_kupci = True, onda rezultat mora da sadrži i polje "ispravka_vrednosti": true i da se u tabelu uključe podaci o umanjenju vrednosti potraživanja (konta tipa 229).

Ako je False, polje "ispravka_vrednosti": false.

Format izlaza:

Vrati JSON u sledećem obliku:

{{
  "file_type": "{file_type}",
  "period_date": "{period_display}",
  "datum": "YYYY-MM-DD",
  "ispravka_vrednosti": {str(is_iv).lower()},
  "tabela": [
    {{
      "konto": "...",
      "naziv partnera": "...",
      "{'promet duguje' if file_type == 'kupci' else 'promet potrazuje'}": ...,
      "saldo": ...
    }},
    ...
  ]
}}

ULAZNI FAJL:
{extracted_text}

Vrati SAMO JSON bez dodatnih objašnjenja."""
        
        return prompt
    
    async def process_extracted_text(
        self,
        extracted_text: str,
        file_type: str,
        period: str,
        is_iv: bool = False
    ) -> Dict[str, Any]:
        """Process extracted text using custom AI prompt"""
        
        if not self.client:
            raise ValueError("Azure AI client not initialized")
        
        try:
            prompt = self._get_custom_prompt(file_type, period, is_iv, extracted_text)
            
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = self.client.complete(
                messages=messages,
                model=settings.azure_ai_foundry_model,
                temperature=0.1,  # Low temperature for consistent output
                max_tokens=4000
            )
            
            # Extract JSON from response
            response_content = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                # Clean response - sometimes AI returns extra text
                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_content = response_content[json_start:json_end]
                    result = json.loads(json_content)
                    
                    logger.info(f"Successfully processed {file_type} file for {period}")
                    return result
                else:
                    raise ValueError("No valid JSON found in AI response")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI response as JSON: {e}")
                logger.error(f"AI Response: {response_content}")
                raise ValueError(f"Invalid JSON response from AI: {e}")
            
        except AzureError as e:
            logger.error(f"Azure AI error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing text with AI: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if AI service is available"""
        return self.client is not None