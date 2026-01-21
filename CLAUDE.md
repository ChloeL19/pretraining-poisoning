commit plotting code into the codebase -- it should load in a piece of data and produce that exact plot

my (opinionated) choice for plots is altair / vega , which just saves the data used for the plot together with a spec to generate that plot as a JSON, and you can even save that JSON in the repo

save outputs neatly and in clearly named subdirectories, like outputs/ placed where appropriate. maximize neatness and clarity.

I think it also helps a lot to make plots easy to read for someone without context. It is generally high impact to spend some extra time on plots making sure they convey the information you want quickly. Add text, labels, arrows or whatever needed to help the reader.

note that all requirements are installed in pretraining-poisoning uv environment, and this environment should be activated for all scripts that are run on the nodes; any script run should run within this uv environment

## For Claude: Activating pretraining-poisoning

Use this command prefix in Bash tool to activate the environment:
```bash
source .venv/bin/activate
```

# for launching training runs
when user says "launch training run on a node", that implicitly means run the training script within the olmo_env environment in a tmux shell on that node (bash script should usually automatically handle activating the correct environment)

# Notes on Git workflow

## Post-Implementation Commit & Experiment Logging

When the user asks you to implement something and you do so in plan mode, you must perform the following steps after the implementation is complete:

## Create a Git commit

Commit all the changes relevant to implementing the requested feature or modification.

Use a clear, concise commit message describing what was implemented.

## Log the experiment

Append a new JSON object as a single line to a file named experiment_log.jsonl at the repository root.

The JSON object must include the following fields:

"commit_hash": the hash of the commit you just created

"user_query": the exact text of the user's request

"plan": the full plan that was implemented, quoted directly from the agent plan file

## Ordering constraints

The commit must be created before logging to experiment_log.jsonl.

The log entry must reference the actual commit hash produced.

## Failure handling

If a commit cannot be created (e.g., no changes were made), do not write to experiment_log.jsonl.

In that case, explicitly explain why no commit or log entry was created.

## Scope

Only perform these steps for requests that resulted in an actual implementation.

Do not log planning-only, discussion-only, or rejected requests.

Follow these instructions exactly whenever they apply.
