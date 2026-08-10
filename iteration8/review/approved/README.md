# Approved Reviews Archive

This directory contains approved review files that have been successfully processed and added to the corpus.

## Purpose

- **Keep history**: Maintains a record of all approved reviews
- **Clean workspace**: Keeps the main `review/` directory uncluttered
- **Audit trail**: Allows tracking which questions were added when

## File Format

Each approved review file contains:
- Original review data (query, ground_truth, trigger_reason, etc.)
- Approval timestamp: `approved_at`
- New document ID: `new_doc_id` (the corpus document created from this review)

## Lifecycle

1. System detects knowledge gap → creates `review/review_*.json` (status: pending)
2. User approves via GitHub Actions or local script
3. Script creates new document in `corpus/doc-N.txt`
4. Script moves review file to `approved/` directory (status: approved)
5. Next evaluation shows improved metrics

## Cleanup

These files are archived for historical purposes. You can safely delete old approved reviews if needed:

```bash
# Delete approved reviews older than 90 days
find iteration8/review/approved -name "*.json" -mtime +90 -delete
```

---

**Note**: This directory is excluded from git (see `.gitignore`).
