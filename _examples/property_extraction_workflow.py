#!/usr/bin/env python3
"""
Property Extraction Workflow Example

This script demonstrates the complete workflow for extracting structured
Italian property auction data from PDF appraisal documents.

Workflow:
1. Select a deployed GPU OCR endpoint (OLM, RoLM, or Mistral)
2. Extract text from PDF using OCR
3. Convert unstructured text to structured AuctionPropertyProfile JSON

Usage:
    python property_extraction_workflow.py <pdf_file> [--provider olm_ocr] [--output output.json]
"""

import asyncio
import argparse
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocular.services.property_extraction_service import (
    PropertyExtractionService,
    extract_property_from_pdf
)
from ocular.models.property_assessment import AuctionPropertyProfile


async def example_basic_extraction(pdf_path: str, provider: str = "olm_ocr"):
    """
    Basic example: Extract property data from a single PDF.

    Args:
        pdf_path: Path to the property appraisal PDF
        provider: OCR provider to use
    """
    print(f"📄 Processing: {pdf_path}")
    print(f"🤖 Using provider: {provider}")
    print("-" * 60)

    try:
        # Quick extraction using convenience function
        property_data = await extract_property_from_pdf(
            pdf_path=pdf_path,
            provider=provider,
            save_output=True
        )

        print("\n✅ Extraction completed successfully!")
        print(f"\n📊 Property Details:")
        print(f"   Sale ID: {property_data.sale_id}")
        print(f"   Lot Number: {property_data.lot_number}")
        print(f"   Court: {property_data.appraisal_summary.court}")
        print(f"   Base Price: €{property_data.auction_details.base_price:,.2f}")
        print(f"   Number of Assets: {len(property_data.assets)}")

        print(f"\n🏠 Assets:")
        for idx, asset in enumerate(property_data.assets, 1):
            print(f"   {idx}. {asset.asset_type} - {asset.surface_sqm}m²")
            print(f"      Address: {asset.address}")

        print(f"\n⚠️  Compliance Issues: {len(property_data.risk_assessment.compliance_issues)}")
        for issue in property_data.risk_assessment.compliance_issues:
            print(f"   - {issue}")

        return property_data

    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        raise


async def example_full_workflow(
    pdf_path: str,
    provider: str = "olm_ocr",
    deployment_id: str = None,
    additional_context: dict = None,
    output_file: str = None
):
    """
    Complete workflow example with all options.

    Args:
        pdf_path: Path to the property appraisal PDF
        provider: OCR provider to use
        deployment_id: Optional RunPod deployment ID
        additional_context: Optional portal JSON data
        output_file: Path to save output JSON
    """
    print("="*60)
    print("🏛️  PROPERTY EXTRACTION WORKFLOW")
    print("="*60)

    # Initialize service
    service = PropertyExtractionService()

    # Step 1: Select OCR endpoint
    print("\n📡 Step 1: Selecting OCR endpoint...")
    ocr_provider = await service.select_ocr_endpoint(
        preferred_provider=provider,
        deployment_id=deployment_id
    )
    print(f"   Selected: {ocr_provider.provider_name}")

    # Step 2: Extract text
    print("\n📄 Step 2: Extracting text from PDF...")
    raw_text = await service.extract_text_from_pdf(
        pdf_path=pdf_path,
        provider=ocr_provider
    )
    print(f"   Extracted: {len(raw_text)} characters")
    print(f"   Preview: {raw_text[:200]}...")

    # Step 3: Convert to structured JSON
    print("\n🔄 Step 3: Converting to structured JSON...")
    structured_data = await service.convert_to_structured_json(
        extracted_text=raw_text,
        additional_context=additional_context
    )
    print(f"   ✅ Conversion successful!")

    # Step 4: Process complete document (all-in-one)
    print("\n🚀 Step 4: Running complete workflow...")
    result = await service.process_property_appraisal(
        pdf_path=pdf_path,
        provider_name=provider,
        deployment_id=deployment_id,
        additional_context=additional_context,
        save_output=bool(output_file),
        output_dir=str(Path(output_file).parent) if output_file else None
    )

    print(f"\n✅ Complete workflow finished!")
    print(f"   Processing time: {result['metadata']['processing_time_seconds']:.2f}s")
    print(f"   Provider: {result['metadata']['provider']}")

    # Display results
    property_profile = AuctionPropertyProfile(**result["structured_data"])
    print(f"\n📊 EXTRACTED PROPERTY DATA:")
    print(f"   Sale ID: {property_profile.sale_id}")
    print(f"   Lot: {property_profile.lot_number}")
    print(f"   Court: {property_profile.appraisal_summary.court}")
    print(f"   Procedure: {property_profile.appraisal_summary.procedure_number}")
    print(f"   Base Price: €{property_profile.auction_details.base_price:,.2f}")
    print(f"   Min Bid: €{property_profile.auction_details.minimum_bid:,.2f}")
    print(f"   Auction Date: {property_profile.auction_details.auction_date}")

    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n💾 Output saved to: {output_path}")

    return result


async def example_batch_extraction(pdf_files: list, provider: str = "olm_ocr"):
    """
    Batch extraction example: Process multiple PDFs.

    Args:
        pdf_files: List of PDF file paths
        provider: OCR provider to use
    """
    print(f"📚 Processing {len(pdf_files)} files in batch...")
    print("-" * 60)

    service = PropertyExtractionService()
    results = []

    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {Path(pdf_path).name}")

        try:
            result = await service.process_property_appraisal(
                pdf_path=pdf_path,
                provider_name=provider,
                save_output=True
            )

            results.append({
                "file": pdf_path,
                "success": True,
                "sale_id": result["structured_data"]["sale_id"],
                "lot_number": result["structured_data"]["lot_number"]
            })

            print(f"   ✅ Success - Sale ID: {result['structured_data']['sale_id']}")

        except Exception as e:
            results.append({
                "file": pdf_path,
                "success": False,
                "error": str(e)
            })
            print(f"   ❌ Failed: {e}")

    # Summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print(f"\n{'='*60}")
    print(f"📊 BATCH PROCESSING SUMMARY:")
    print(f"   Total: {len(results)}")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"{'='*60}")

    return results


async def example_with_portal_data(pdf_path: str, portal_json_path: str):
    """
    Example: Combine PDF appraisal with portal JSON data.

    Args:
        pdf_path: Path to the property appraisal PDF
        portal_json_path: Path to JSON data from auction portal
    """
    print("🔗 Combining PDF appraisal with portal data...")

    # Load portal data
    with open(portal_json_path, 'r', encoding='utf-8') as f:
        portal_data = json.load(f)

    print(f"   Portal data loaded: Sale ID {portal_data.get('idVendita')}")

    # Extract with additional context
    service = PropertyExtractionService()
    result = await service.process_property_appraisal(
        pdf_path=pdf_path,
        additional_context=portal_data,
        save_output=True
    )

    print(f"\n✅ Extraction with portal context completed!")
    print(f"   Combined data for Sale ID: {result['structured_data']['sale_id']}")

    return result


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract structured property auction data from PDF appraisals"
    )
    parser.add_argument(
        "pdf_file",
        help="Path to the property appraisal PDF file"
    )
    parser.add_argument(
        "--provider",
        default="olm_ocr",
        choices=["olm_ocr", "rolm_ocr", "mistral", "google_vision", "aws_textract"],
        help="OCR provider to use (default: olm_ocr)"
    )
    parser.add_argument(
        "--deployment-id",
        help="RunPod deployment ID (for OLM/RoLM providers)"
    )
    parser.add_argument(
        "--portal-json",
        help="Path to JSON file with additional auction portal data"
    )
    parser.add_argument(
        "--output",
        help="Path to save the output JSON file"
    )
    parser.add_argument(
        "--batch",
        nargs="+",
        help="Process multiple PDFs in batch mode"
    )

    args = parser.parse_args()

    # Load portal data if provided
    portal_data = None
    if args.portal_json:
        with open(args.portal_json, 'r', encoding='utf-8') as f:
            portal_data = json.load(f)

    # Run appropriate workflow
    if args.batch:
        # Batch mode
        asyncio.run(example_batch_extraction(args.batch, args.provider))
    else:
        # Single file mode
        asyncio.run(example_full_workflow(
            pdf_path=args.pdf_file,
            provider=args.provider,
            deployment_id=args.deployment_id,
            additional_context=portal_data,
            output_file=args.output
        ))


if __name__ == "__main__":
    main()
