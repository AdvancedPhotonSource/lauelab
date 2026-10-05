# Documentation writing style

## Purpose

This profile defines the writing style for the `lauelab` documentation. Use it when writing or reviewing narrative pages, examples, API introductions, and supporting reference text.

The profile determines how two general editing methods apply here — a general editorial pass that removes filler and generated-text patterns, and Simplified Technical English (ASD-STE100) for instructions that must not be misread — and records conventions specific to this project.

The goals are scientific accuracy, direct instruction, and consistent terminology. Clarity takes priority over brevity. Do not simplify a statement if the shorter form changes its scientific meaning or removes a required qualification.

## Audience

Assume that readers:

- Know the basics of X-ray science and Laue diffraction.
- Can use Python and NumPy for scientific work.
- May be new to `lauelab` and the 34-ID-E acquisition conventions.
- Need enough context to choose inputs, interpret results, and diagnose common failures.

Do not teach introductory crystallography or basic Python on the main documentation path. Define package-specific terms and conventions even when the underlying scientific concept is familiar.

Link to deeper material when a correct explanation would interrupt the task at hand.

## Voice and tone

Address the reader as "you" when doing so makes an instruction clearer.

Use the imperative for procedures:

> Load the geometry once and reuse the `Indexer` for each frame.

Use direct descriptive prose for behavior:

> `index_many` returns one `FrameResult` for each input frame.

Use first-person plural only when it identifies a deliberate project decision. Prefer a direct statement in most cases. Do not use first-person singular in project documentation.

Keep the tone technical and approachable. Do not use promotional claims, conversational filler, or artificial enthusiasm. Recommendations must include a scientific or operational reason.

Prefer active voice, but use passive voice when the actor is unknown or irrelevant:

> Peaks below the threshold are discarded.

Vary sentence length naturally. Split a sentence when a reader must backtrack to parse it. Do not enforce a fixed word limit on explanatory prose.

Use sentence case for headings.

### State facts plainly

Write what is stored where and what the code does. Do not give files, datasets, detectors, or results agency through figurative verbs, and do not compress an explanation into a catchy phrase that the reader must decode.

Only software that performs an operation can be the subject of an action verb, and the verb must be literal: a function *returns*, *reads*, *writes*, or *raises*; the indexer *reads* a frame. Data does not act. For data, use "contains", "is stored in", "is in", or the passive voice.

| Avoid | Write |
| --- | --- |
| What the detector writes | Input file layout |
| The catalog holds the run settings once. | `scan.h5` contains the reconstruction settings, which are the same for all points. |
| Three conventions decide which slices are used. | Three conventions of the file format determine which slices are used. |
| The read rate sets most of the run time. | Most of the run time is spent reading the input file. |
| The result keeps the reference. | The result contains the `ScanFrame`. |
| Nothing that `h5py` cannot see | The reader classes read the same datasets as `h5py`. |
| Depth traces over regions of interest | The summed intensity in a small detector region at each depth |
| Pixels never leave the worker. | The worker writes the frames to the point file and returns only the status. |

Avoid these related patterns:

- Headings phrased as a teaser ("What the detector writes"). Use a plain description of the content ("Input file layout").
- A bold lead-in followed by a dash and a gloss. Write a complete sentence.
- Aphorisms and contrast for effect ("They record, but do not prove, which input was used"). State the limitation directly: "They do not guarantee that the input file was unchanged."
- Noun stacks used as shorthand for a quantity. Name the quantity and its reduction: what is summed, over which pixels, at which depths.

This applies to code comments, printed messages, plot titles, and notebook prose as well as to the documentation pages.

## Teaching approach

Organize each page around a reader outcome. Explain enough context for the reader to complete the task and interpret the result.

For a new concept, use this order when practical:

1. State what the concept represents.
2. Explain why the reader needs it.
3. Show the relevant API or data shape.
4. Give a small example.
5. Describe constraints and common mistakes.
6. Link to deeper scientific or API material.

Keep the main path operational. Put detailed derivations, numerical assumptions, and algorithm discussions deeper in the Concepts section.

Do not repeat the full API reference in a guide. Explain how and why to use an object, then link to its reference entry for the complete signature and field list.

## Terminology

Use one term for one project concept. Do not rotate synonyms for variety.

Use these forms consistently:

- X-ray
- Laue diffraction
- frame
- detector
- detector slot
- peak search
- pixel-to-q conversion
- crystal indexing
- scattering vector
- geometry file
- crystal description
- `Indexer`
- `FrameResult`
- `Pattern`

Use exact Python names in code formatting. Define domain terms when their meaning in this package is narrower than their general scientific meaning.

Do not replace valid mathematical or software terms merely because they appear on a general jargon list. Words such as "vector" and "primitive" are appropriate when they carry their precise technical meaning.

## Units, shapes, and notation

Use a hybrid of scientific notation and code notation.

### Scientific quantities

- Put a space between a number and its unit: 10 mm, 25 µm, 12 keV.
- Use consistent unit symbols: mm, µm, rad, deg, and keV.
- State the unit whenever a value is not dimensionless.
- Include a **Units** column in parameter tables when parameters have mixed units.
- Use rendered mathematics for equations and scientific relationships.
- Define every symbol near the equation where it first appears.
- Do not use an equation when prose or a small table communicates the same information more clearly.

### Python values and arrays

- Put Python names, values, dtypes, and literal shapes in code formatting.
- Write an image array shape as `(ny, nx)`.
- Write an image coordinate as `(x, y)`.
- State explicitly that `image[y, x]` accesses the pixel at `(x, y)`.
- Write tabular array shapes as `(n, 2)`, `(n, 3)`, and similar forms.
- State relevant dtypes, such as `numpy.uint16` or `numpy.float64`.
- State whether an index or coordinate is zero-based.
- Distinguish pixel coordinates from NumPy indexing whenever both appear in one section.

Use tables for comparable fields such as parameter defaults, units, constraints, and effects. Use prose when entries require qualifications that a compact table would hide.

## Coordinate conventions

Use the 34-ID-E conventions implemented by `lauelab`. This is the correct initial scope because the package, geometry format, and native indexing implementation originate from the 34-ID-E workflow.

Do not refer to a single undifferentiated "34-ID-E coordinate system." Identify the coordinate space whenever values cross an API boundary or a transformation is discussed.

Document these spaces separately:

- **NumPy array indices:** `[y, x]`, with array shape `(ny, nx)`.
- **Frame pixel coordinates:** zero-based `(x, y)` coordinates in the supplied frame.
- **Full-detector pixel coordinates:** frame coordinates transformed by `start` and `group`.
- **Detector-local physical coordinates:** positions derived from detector pixel dimensions and physical size.
- **34-ID-E laboratory coordinates:** positions and directions produced from geometry rotations and translations.
- **Scattering-vector coordinates:** `qhat` components in the 34-ID-E laboratory frame.
- **Sample positions:** `(x, y, z)` values in the acquisition coordinate system.
- **Crystal coordinates:** fractional atom positions and the documented crystal-basis convention.

For grouped data, explain that the conversion maps a frame coordinate to the center of its corresponding full-detector pixel group.

Treat detector indices as physical geometry slots, not ordinal positions among active detectors. Geometry slots can be sparse. This distinction requires a warning where detector selection is introduced.

Do not infer names such as "vertical," "outboard," "upstream," or "downstream" from the source code. A 34-ID-E domain source or beamline expert must verify physical axis directions and handedness before those labels enter published documentation.

## Notes, tips, and warnings

Use a moderate admonition policy.

### Note

Use a note for context that helps interpretation but is not required to complete the current task.

Appropriate subjects include compatibility information, a useful default, or a link between related concepts.

### Tip

Use a tip for optional workflow advice that saves time or reduces repeated work.

A tip must identify a concrete action and benefit. Do not use tips for generic encouragement.

### Warning

Use a warning when an apparently valid action can produce an incorrect scientific result, select the wrong data, lose data, or cause a costly operation.

Use warnings rarely. Invalid input that immediately raises a clear exception usually belongs in ordinary error documentation, not in a warning.

Sparse detector-slot semantics require a warning because a mistaken assumption can select a different physical detector.

### Implementation details

Include an implementation detail only when it explains observable behavior, memory ownership, numerical behavior, performance, file compatibility, or a limitation that affects correct use.

Present a short implementation detail as ordinary prose. Use a clearly named subsection when the explanation is substantial. Do not expose native function names, private helpers, or internal data structures unless a contributor page specifically discusses them.

## Mathematical and algorithmic depth

Keep equations and algorithm details away from the initial user path unless they are necessary to use the API correctly.

The landing page, installation guide, and quickstart should contain little or no mathematics. Task-oriented guides can include a small equation when it is needed to define an input, output, unit, or parameter effect.

Place deeper material under Concepts in dedicated algorithm pages:

- Peak search
- Pixel-to-q conversion
- Crystal indexing
- Depth reconstruction

An algorithm page can include coordinate transforms, scattering-vector definitions, fitting and matching concepts, numerical assumptions, limitations, and primary references.

Each equation or implementation explanation must help a reader do at least one of the following:

- Interpret an output.
- Configure an operation.
- Verify a result.
- Understand a documented limitation.

Do not add mathematical detail only to make a page appear complete. Mark scientific claims and derivations for domain review when the implementation is not sufficient evidence.

## Examples

Use synthetic data for the initial documentation. Synthetic examples make the required inputs explicit and keep the documentation reproducible.

Add beamline examples later when reviewed data and usage rights are available. Clearly label synthetic and measured data so readers do not mistake one for the other.

Examples must:

- Use the supported public API.
- Include all imports needed for the shown code.
- Use deterministic inputs where practical.
- State important shapes, dtypes, and units.
- Avoid unexplained setup unrelated to the lesson.
- Show a useful result, not only that the call completed.
- Handle expected failure when error behavior is the subject.

Execute pure-Python examples where practical. Validate native indexing examples with integration tests. The basic documentation build must not require native indexing execution.

Do not invent temporary visualization code to fill plotting gaps. Add plotting examples after the public visualization API is established.

## Guidance by page type

### Landing page

State what the package does, who it is for, and where a new reader should begin. Keep details and mathematics out of this page. Avoid claims about speed, robustness, or scientific quality unless measurements support them.

### Installation

Use short procedures and explicit prerequisites. State supported platforms and native dependencies precisely. Apply the strictest structural clarity rules on this page.

### Quickstart

Show the shortest complete indexing workflow. Assume the reader already has the required scientific background. Explain only the inputs and outputs needed for the example, then link to the detailed guides.

### Concepts

Explain the model that connects inputs, processing stages, and outputs. Use selective mathematics. Separate an operational overview from detailed algorithm pages.

### User guides

Start from a task or decision. Explain parameter effects and scientific consequences, not only types and accepted values. Include common mistakes that are plausible and consequential.

### API reference

Keep descriptions exact and compact. State types, shapes, units, defaults, exceptions, ownership, and observable behavior. Do not use reference entries as tutorials.

### Error documentation

State what failed, why it can fail, and what the reader can check. Preserve uncertainty when several causes are possible. Do not imply that retrying is safe unless the behavior supports that advice.

## Applying the editing skills

Apply the general editorial pass with these project-specific exceptions:

- Do not add personal opinions or first-person singular voice.
- Do not introduce deliberate messiness. Natural rhythm is useful, but scientific prose must remain deliberate.
- Parentheses are permitted for units, abbreviations, mathematical qualifications, and short technical clarification.
- Keep precise mathematical and software terminology.
- Keep meaningful scientific adverbs.
- Use passive voice when the actor does not matter.

Apply Simplified Technical English (ASD-STE100) selectively.

Use strict structural discipline for:

- Procedures
- Warnings
- Input constraints
- Error recovery
- Parameter requirements
- Statements whose misreading could change a scientific result

Use STE-flavored guidance for explanatory prose:

- Prefer active voice.
- Keep terminology consistent.
- Define domain terms.
- Prefer direct verbs over nominalizations.
- Use lists for sequences and multiple conditions.
- Preserve qualifications, uncertainty, and scope.

Do not enforce strict ASD-STE100 sentence limits, noun-cluster limits, tense restrictions, or the blanket phrasal-verb ban on scientific narrative. Treat sentence length as a review signal rather than a compliance target.

Do not claim ASD-STE100 compliance. The project uses selected principles for clarity and does not validate text against the official controlled vocabulary.

## Review checklist

Before a narrative page is complete, verify that:

- The page has a clear reader outcome.
- The assumed background matches this profile.
- Project terms remain consistent.
- Every shape, dtype, unit, and coordinate space is explicit where relevant.
- Pixel `(x, y)` and array `[y, x]` conventions cannot be confused.
- Scientific claims follow from reviewed sources, tests, or implementation behavior.
- Recommendations include their reason and scope.
- Warnings identify a consequential and plausible mistake.
- Examples use public APIs and have an identified validation method.
- Deep algorithm material does not interrupt the main task.
- Links replace unnecessary duplication.
- The text contains no promotional filler, chatbot phrasing, or generic conclusion.
- Data, files, and devices are not given figurative actions ("the catalog holds", "the detector writes"), and no heading or phrase needs decoding.
- Simplification has not removed uncertainty or scientific precision.

## Items that require domain review

The following details remain open until a 34-ID-E source or beamline expert verifies them:

- Physical names and positive directions of the laboratory axes.
- Laboratory-frame handedness as it should be explained to users.
- The physical sign convention for sample depth.
- The complete relationship among detector-local, laboratory, scattering-vector, and crystal coordinates.
- Scientific parameter-tuning recommendations.
- Algorithm derivations or quality claims not established by tests or primary references.

Until review is complete, describe only verified API behavior and transformations. Do not infer physical interpretations from variable names alone.
