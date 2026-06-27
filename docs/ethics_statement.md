# P01 — Sign Language Translation · Ethics Statement

**Author:** Le Dinh Minh Quan (23127460) · NLP in Industry, Final Assignment.
**Package:** `signlang` · **Folder:** `01_Sign_Language_Translation/`.

This document states the ethical commitments and constraints governing the P01 Sign Language Translation (SLT)
system. P01 maps sign-language **video / pose-keypoint sequences → spoken-language text** via an intermediate
**gloss** sequence (the Sign2Gloss2Text cascade: MediaPipe Holistic pose extraction → motion-based segmentation →
per-segment gloss recognition → gloss→text translation). Because sign language is the native language of Deaf
communities and the input is **biometric video of a person's face and hands**, the ethical stakes here are higher
and more specific than for the prior 18 text/OCR projects. This statement is binding on the design, the deployment,
and any downstream use. It is not boilerplate; it is scoped to what this system actually does and the specific
people it touches.

---

## 1. Headline position: assistive tooling, NOT a replacement for human interpreters

**State this plainly and without hedging: P01 is an assistive aid. It is NOT, and must never be presented as, a
substitute for a qualified human sign-language interpreter.**

Professional Deaf and hearing interpreters are trained, certified, accountable, and able to handle the full
linguistic, cultural, and pragmatic richness of a live conversation — register, classifiers, role-shifting,
fingerspelling, regional variation, repair when something is unclear, and the cultural mediation that an interpreting
encounter requires. P01 does none of this. In its honest, fully-offline configuration it recognizes a **40-gloss
synthetic vocabulary** and translates it through a fixed lexicon; even the upgraded Colab path (a ~5M-parameter pose
transformer + `t5-small` translator) is a narrow, small-vocabulary research prototype trained largely on synthetic
data, because — as the design brief documents — **there is no permissively-licensed, directly-loadable continuous-SLT
corpus or Sign→Text checkpoint on the Hugging Face Hub** (every continuous benchmark is non-commercial, gated, or
unspecified). A system built on this foundation is, by construction, far from interpreter-grade.

Concretely, the following are **out-of-scope and prohibited uses** of P01:

- As the **sole** communication channel in any **medical** setting (diagnosis, consent, triage, mental-health,
  emergency care).
- As the **sole** channel in any **legal** setting (police interaction, arrest, court testimony, deposition,
  legal-advice meetings, immigration interviews).
- In any setting where a wrong or missing translation could affect a person's **safety, liberty, health, finances,
  legal rights, or access to services**.

In all such settings a qualified human interpreter is required; P01, at most, is a convenience aid for low-stakes,
informal exchanges where the user understands its limits. The right to a qualified interpreter (recognized in many
jurisdictions, e.g. accessibility and anti-discrimination law) is **not** discharged by deploying this software.

## 2. Representation and linguistic bias — the central technical-ethical risk

Sign languages are **full, natural human languages** with their own grammar, phonology, and lexicon — not gestural
codes for spoken language, and not a single universal "sign language."

- **ASL ≠ BSL ≠ signed-English ≠ ISL ≠ Auslan ≠ VGS …** American Sign Language and British Sign Language are mutually
  unintelligible and historically unrelated; both differ from manually-coded *signed* forms of a spoken language
  (e.g. Signing Exact English), which are not natural sign languages at all. There are well over a hundred distinct
  sign languages worldwide.
- **Within one sign language**, there is heavy variation across **region, age, race (e.g. Black ASL), gender, Deaf
  vs hard-of-hearing community, and individual signing style** (signing speed, signing space, handshape precision,
  one- vs two-handed variants).

A model trained on one sign language and one signer set will **fail — often silently and confidently — on others.**
P01 inherits and amplifies this in specific, nameable ways:

- The offline pipeline is trained on a **synthetic generator of 40 ASL-style glosses** with deterministic motion
  trajectories. It encodes *no* real signer diversity, *no* regional or sociolinguistic variation, and *no* sign
  language other than the toy ASL-style vocabulary it was seeded with.
- The only cleanly-permissive *real* corpus available for the smoke test is
  `Sigurdur/icelandic-sign-language` (Apache-2.0, **214 rows**, a YouTube-SL-25 Icelandic slice) — a different sign
  language and a tiny, narrow sample.
- The reference and would-be training corpora are restrictively licensed and themselves skewed: How2Sign / iSign /
  WLASL each reflect a particular language (ASL, Indian SL), a particular set of signers, and particular recording
  conditions.

**Commitment:** we treat signer-independence and cross-sign-language generalization as **unsolved** for this system.
We will not market or describe P01 as supporting any sign language it was not explicitly evaluated on, and we will
not present per-signer performance as representative. The model card and `privacy_robustness` doc must name the exact
language(s) and data the system was tested on, and state that **performance on any other signer or sign language is
unknown and presumed poor.** Performance disparities across signer subgroups are a fairness defect to be measured and
reported, not an afterthought — but with the synthetic spine we make no claim that such fairness has been achieved.

## 3. Biometric-surveillance risk of capturing signers

To translate signing, P01 must capture **video of a person's hands, body, and — via MediaPipe Holistic — face
landmarks.** This is **biometric and directly identifying** data of a kind that is especially sensitive because it
belongs to a **minority community that has historically been surveilled, pathologized, and denied linguistic rights.**

Specific risks we recognize and design against:

- **Identification and re-identification.** Face/hand geometry and idiosyncratic signing style can identify an
  individual. Pose-keypoint sequences are *not* anonymous: gait/sign-style and facial-landmark data are biometric.
- **Function creep into surveillance.** A pipeline that detects and transcribes signing could be repurposed to
  *monitor* who is signing, where, to whom, and about what — a chilling, rights-violating use against a community
  for whom signing in public is simply speaking their language.
- **Content sensitivity.** Captured signing may include private, medical, legal, or intimate content.

**Design commitments (mirrored in `privacy_robustness` and the deployment docs):**

- **Data minimization & no retention by default.** The system does not store raw video or pose sequences by default.
  The pose front-end extracts landmarks and discards frames; outputs are returned, not logged with identifying input.
- **On-device / edge-first processing.** The fully-offline path (`SeedPoseEngine` + numpy nearest-centroid + lexicon)
  runs on CPU with **no network and no cloud dependency**, so signing video need never leave the user's device.
- **Optional LLM "brain" OFF by default.** The advisory `anthropic` LLM hook in the agent is disabled by default,
  never changes the output, and — critically — means **no signing data is sent to a third-party API** in the default
  configuration. Enabling it is an explicit, documented choice with its own consent implications.
- **No covert capture.** P01 must only run on video the signer knowingly provides for translation. Using it to
  detect or transcribe signing without the signer's awareness and consent is a prohibited use.
- **Consent is informed and revocable**, and covers what is captured, where it is processed, whether anything is
  retained, and for what purpose.

## 4. False translations in high-stakes settings → abstention, per-sign confidence, human-in-the-loop

The most dangerous failure mode of an SLT system is **not** breaking — it is **confidently emitting a fluent, wrong
translation.** A blind end-to-end decoder will happily hallucinate grammatical spoken text from noise, low-quality
video, or out-of-vocabulary signing. In a medical or legal setting, a plausible-but-wrong sentence is worse than a
visible "I don't know," because a hearing party may act on it as if it were accurate.

P01's agent is built specifically to **refuse to do this.** The deterministic 5-decision FSM
(`src/signlang/agent/`) makes uncertainty explicit at every step:

- **D1 ingest** — gate on frame count; reject inputs too short to translate rather than guessing.
- **D2 segment** — motion-based segmentation into individual sign units (low-velocity rest frames split signs),
  so confidence is assessed **per sign**, not smeared across a whole utterance.
- **D3 recognize** — emit a per-segment gloss **with a confidence**; anything below `recog_min_conf = 0.15` is
  flagged **low-confidence** rather than silently accepted.
- **D4 translate + verify** — a round-trip text→gloss agreement check / chrF keep-gate that drops translations the
  system cannot self-verify, instead of shipping them.
- **D5 finalize — ABSTENTION.** If the low-confidence / out-of-vocabulary segment ratio exceeds
  `oov_abstain_ratio = 0.5`, the system returns **"uncertain" + `needs_review`** instead of a fabricated sentence.

We verified offline that **pure-noise input abstains** rather than hallucinating, and that all five decision points
fire. This abstention behavior is the system's core ethical value-add: **it is designed to be silent when it should
be silent.** In any setting that matters, abstention or low confidence must trigger a **human in the loop** — a
qualified interpreter or a Deaf participant who can confirm or correct — and the output must never be auto-actioned.

**We also publish an honesty caveat on the metrics themselves.** Automatic SLT metrics (BLEU, chrF, ROUGE, BLEURT)
are **unreliable** — length-sensitive and blind to hallucination and semantic equivalence (Yazdani et al., *A
Critical Study of Automatic Evaluation in SLT*, 2025, hf.co/papers/2510.25434). Our headline offline numbers (gloss
accuracy 1.0, BLEU 99+, segmentation-F1 1.0) are on **clean synthetic data** and **must not** be read as evidence of
real-world readiness. The `translation_evaluation` doc reports the standard metric set **and** flags this limitation
prominently. A high BLEU score is not a license to trust the system with a real conversation.

## 5. Deaf-community involvement: co-design, consent, and benefit-sharing

SLT has a documented history of being built **about** Deaf people rather than **with** them — by predominantly
hearing teams, on data collected without meaningful consent or benefit to signers, and shipped as products that
overpromise. P01 is a student project, but it commits to the same principle that should govern any real deployment:
**"Nothing about us without us."**

- **Co-design, not extraction.** Any move of P01 from synthetic prototype toward real use must involve Deaf signers,
  Deaf-led organizations, and qualified interpreters as **partners** in setting requirements, vocabulary, evaluation
  criteria, and acceptable-use boundaries — not merely as data sources or test subjects.
- **Consent and provenance for data.** This system deliberately runs on a **synthetic generator** as its primary data
  precisely because the real corpora are restrictively licensed and their consent provenance is often unclear. We do
  **not** repackage or redistribute gated / non-commercial corpora (iSign is CC-BY-NC-SA **and gated**; How2Sign and
  its Holistic derivatives are CC-BY-NC upstream; WLASL is research-terms "other"). Their licenses are respected and
  flagged, and the academic-server datasets (RWTH-PHOENIX-2014T, CSL-Daily, YouTube-ASL/SL-25) are **not** mirrored.
- **Benefit-sharing.** The point of accessibility technology is to benefit the community whose data and language make
  it possible. Real deployment should return value to Deaf communities (e.g. accessible tooling, employment of Deaf
  experts, open and permissive artifacts) rather than extracting linguistic data for commercial gain.
- **Respect for the language and culture.** Deaf culture treats sign language as central to identity. P01 is framed
  as an *accessibility aid that augments* communication, **never** as something that makes signing or interpreters
  obsolete, and never with the deficit framing that treats Deafness as a problem to be engineered away.

## 6. Transparency

Users on **both** sides of a P01-mediated exchange must be able to see what the system is and is not confident about.
Opaque, take-it-or-leave-it output is unsafe for a system that can be confidently wrong.

P01 exposes, by design:

- **The intermediate glosses**, not just the final spoken sentence — so the recognized signs are inspectable and a
  signer or interpreter can see *what the system thought it saw*.
- **Per-sign confidence** — surfaced through the API (`POST /translate` returns glosses + spoken text + per-sign
  confidence + abstain flag) and the Gradio demo, so low-confidence signs are visibly marked rather than buried.
- **The abstain flag / `needs_review` state** — the system says "uncertain" out loud instead of fabricating.
- **Honest documentation** — the data card, model card, and evaluation doc state the synthetic-data basis, the tiny
  real-data smoke test, the single-language scope, the metric-unreliability caveat, and every restrictive dataset
  license. No capability is claimed that was not measured, and the measured capability is on synthetic data.

## 7. Responsible-use guidance (for any future user or deployer)

**Do:**

- Use P01 only as a **supplementary** aid in **low-stakes, informal** contexts, with all parties aware they are using
  an experimental machine translator.
- Keep a **qualified human interpreter and/or a Deaf participant in the loop**, and treat the human as authoritative
  whenever stakes are non-trivial.
- **Honor abstention and low confidence** — when the system says "uncertain" or marks signs low-confidence, **stop and
  get a human**; do not coax it into a guess.
- Show the **glosses and per-sign confidence** to users; never present only a clean spoken sentence.
- Process on-device, retain nothing by default, and obtain **informed, revocable consent** before capturing anyone's
  signing.
- State plainly the **specific sign language and signer population** the system was evaluated on, and that anything
  outside that is untested.

**Do not:**

- Do **not** use P01 as the sole interpreter in **medical, legal, emergency, employment, financial, or any
  safety-critical** setting.
- Do **not** present its output as authoritative, certified, or interpreter-grade.
- Do **not** use it to **detect, monitor, or transcribe** people's signing without their knowledge and consent
  (surveillance is a prohibited use).
- Do **not** deploy it for a sign language, dialect, or signer population it was not evaluated on, or imply universal
  sign-language support.
- Do **not** redistribute the gated / non-commercial corpora it references, or strip their licenses.
- Do **not** enable the optional LLM brain (which sends data off-device) without re-doing the consent and
  data-handling analysis.

---

**Summary.** P01 is a deliberately honest, narrow research prototype: a synthetic-data-driven SLT cascade whose most
important behavior is **knowing when to abstain.** It is an assistive aid that must keep Deaf people and qualified
interpreters at the center — through co-design, consent, transparency (glosses + per-sign confidence + abstention),
strict privacy (on-device, no-retention, biometric-aware), and explicit refusal to be the last word in any setting
that matters. The technology must never be allowed to speak *for* a Deaf person more confidently than it has earned.
