"""
Structured product database and data fusion helpers with Firestore fallback.
"""

import logging
from typing import Dict, Any, Optional
from config import Config

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIRESTORE_AVAILABLE = True
except Exception:
    FIRESTORE_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProductDB:
    """Product database with optional Firestore backend and in-memory fallback."""

    def __init__(self):
        # Demo dataset; used as fallback if Firestore disabled/unavailable
        self._products: Dict[str, Dict[str, Any]] = {
            "SKU1001": {
                "name": "Acme SmartWatch X",
                "warranty_end": "2026-10-15",
                "return_policy": "30 days from delivery",
                "shipping_sla": "3-5 business days",
                "support_tier": "premium",
            },
            "SKU2002": {
                "name": "Acme Earbuds Pro",
                "warranty_end": "2026-05-01",
                "return_policy": "15 days unopened",
                "shipping_sla": "2-4 business days",
                "support_tier": "standard",
            },
        }
        self._firestore = None
        if Config.FIRESTORE_ENABLED and FIRESTORE_AVAILABLE and Config.FIREBASE_CREDENTIALS:
            try:
                if not firebase_admin._apps:
                    cred = credentials.Certificate(Config.FIREBASE_CREDENTIALS)
                    firebase_admin.initialize_app(cred)
                self._firestore = firestore.client()
                logger.info("Firestore ProductDB initialized")
            except Exception as e:
                logger.warning("Firestore init failed, using in-memory: %s", str(e))

    def get_product(self, sku: str) -> Optional[Dict[str, Any]]:
        sku = (sku or "").strip().upper()
        product = None
        # Try Firestore first
        if self._firestore:
            try:
                doc_ref = self._firestore.collection("products").document(sku)
                doc = doc_ref.get()
                if doc.exists:
                    product = doc.to_dict()
            except Exception as e:
                logger.warning("Firestore get failed for %s: %s", sku, str(e))
        # Fallback to in-memory
        if product is None:
            product = self._products.get(sku)
        if product:
            logger.info("Structured product found: %s", sku)
        return product

    def build_definitive_facts(self, sku: str) -> Optional[Dict[str, Any]]:
        product = self.get_product(sku)
        if not product:
            return None
        return {
            "sku": sku,
            "name": product.get("name"),
            "warranty_end": product.get("warranty_end"),
            "return_policy": product.get("return_policy"),
            "shipping_sla": product.get("shipping_sla"),
            "support_tier": product.get("support_tier"),
        }



