# Model-card publishing conventions

Keep the current card for each public Hugging Face repository in `docs/model-cards/<repository>/README.md` and publish the same text to its Hub README.

Every card uses these sections in order: Overview, Compatibility, Usage, Evaluation, Limitations, Artifacts and provenance, License. Place the creator attribution and shared project links directly below the title. Preserve the Hub YAML metadata.

Bridge examples use `lp.load(repo)`, `source.capture`, `bridge.translate` and `target.session(...).ask`. Compaction examples use `lp.load_model(repo)`, `model.prefill`, `state.compact` and `model.session(...).ask`. Change the selected repository or model, not the operation syntax. Describe boundary and backend requirements in Compatibility.

Use operation names, model identities, immutable repository revisions and artifact SHA-256 fingerprints in public documentation. Internal experiment labels and private research paths do not belong in cards, metadata, examples or runtime identifiers. Do not remove creator attribution or license grants during cleanup.

Preserve each checkpoint's evaluation data and scope. Distinguish structural runtime checks from answer quality, and previous-checkpoint results from current-checkpoint evidence. Archived preview cards must identify their status and point new users to the current API.

When metadata or reference files change, refresh their manifest and checksum receipts in the same Hub commit. Preserve tensor files, base-model revisions and their hashes. Verify the published README against the reviewed source and scan public text for internal labels. Historical commits and pinned downloads remain immutable.
