# RAG Corpus Triage Batch 2 Summary

**Processed Range:** doc_169 through doc_336
**Total Documents:** 168 files
**Output File:** `/Users/eduardolizana/Documents/Github/21-kmc-rag/data/triage_batch_2.json`
**Processing Date:** March 29, 2026

## Metadata Fields Generated

For each file, the triage process identified:

- **include**: "yes", "no", or "maybe" (whether file should be in RAG system)
- **content_type**: "prose", "structured", or "mixed"
- **sensitivity_flag**: "yes" or "no" (confidential/personal data)
- **summary**: 2-3 sentence description of file purpose
- **topic_tags**: 3-5 comma-separated topic classifications
- **audience**: "investors", "internal", "board", "personal", or "technical"

## Content Distribution

### File Types Processed

- **DOCX (Word documents):** Primary format, mostly compliance documents, SOPs, procedures
- **XLSX/XLS (Spreadsheets):** Data files, planning documents, reference materials
- **PDF:** Certificates, compliance documents, test reports
- **CSV:** Data files (mostly excluded due to sensitivity)
- **JSON:** Configuration/data files
- **ODT:** Legacy format documents

### Content Categories

**Compliance & Governance (Primary):** 100+ documents
- Data Protection Toolkit
- CE Marking & Medical Device Regulations (MDR/GMDN)
- GDPR compliance
- Clinical Safety documentation (DTAC)
- Quality Management System (QMS) documents
- SOPs for all business functions

**Engineering & Technical:** 20+ documents
- Fine-tuning job tracking
- Patient verification prompts
- Software architecture & deployment
- Test plans and reports

**Sales & Marketing:** 10+ documents
- Pitch decks and commercial presentations
- Marketing strategies
- Competitive analysis

**Administrative:** 15+ documents
- Visa sponsorship documentation
- Employee records
- Team planning materials

**Data Files (Excluded):** 20+ documents
- Patient lists and personal data
- Passport/ID documents
- Personnel information

## Key Findings

### Inclusion Recommendations

**Include (yes): ~130 documents**
- All compliance and regulatory documents
- All SOPs and quality management materials
- Key strategic and operational documents
- Marketing and sales materials
- Technical documentation

**Maybe (include): ~30 documents**
- Data spreadsheets with reference value
- Template documents
- Archive documents in deprecated folders
- Personal CVs and business plans

**Exclude (no): ~8 documents**
- Patient data files (CSV/XLSX with patient lists)
- Personal identification documents (passports, national IDs)
- Council tax records
- Pure raw data files without narrative value

### Sensitivity Flags

**Flagged (sensitive): ~40 documents**
- Patient Verification data (flagged but potentially valuable)
- Financial forecasts and cap tables
- Personnel records
- Personal identification documents
- Visa/visa sponsorship documentation

Most sensitive content is excluded from RAG per instructions.

## Document Quality Notes

**High-Value Documents for RAG:**
- Summary.md (company overview and tech stack)
- All compliance toolkits and data protection documentation
- Software architecture and technical documentation
- Clinical safety and quality management materials
- Regulatory compliance documentation

**Lower-Value for RAG:**
- Pure data spreadsheets without narrative
- Encrypted or protected PDFs
- Template documents (without implementation examples)
- Deprecated folder contents
- Raw machine learning training data

## Recommendations

1. **Priority for Inclusion:** All compliance, regulatory, and quality management documents - these provide essential business context
2. **Consider Including:** Spreadsheets with competitive analysis and strategic planning
3. **Definitely Exclude:** All personal data, passport/ID scans, and patient lists
4. **Review for Inclusion:** Deprecated folder contents - may be archived but not relevant

## File Statistics

- **Total files processed:** 168
- **Structured (xlsx/csv):** ~50
- **Prose (docx/pdf/md):** ~115
- **Mixed (pptx/odt/json):** ~3

- **Board audience:** ~95 documents
- **Internal audience:** ~60 documents
- **Investor audience:** ~10 documents

- **Compliance/governance tags:** ~100 documents
- **Data/reference tags:** ~30 documents
- **Product/marketing tags:** ~15 documents
- **Operations/process tags:** ~23 documents
