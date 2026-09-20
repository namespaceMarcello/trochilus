# experts_steady.awk — what one GENERATED token costs the expert store, from a
# `experts_budget.sh misses` session. The store's counters cover the prompt too, and the prompt
# touches nearly every expert, so the cost of a token is the DIFFERENCE between a run that
# generates 72 tokens and one that generates 8, over the 64 tokens between them.
#
# Input: the run lines of tools/ab_modes.sh, "<label> <phase> <round> <value>", where the labels
# are <name>short and <name>long. Round 0 is dropped as warm-up, like the medians there.
# Output: one line per budget, misses and MiB per generated token, and what the prompt alone cost.
$3 > 0 && ($2 == "misses" || $2 == "mib") {
  label = $1
  kind = ""
  if (label ~ /short$/) { kind = "short"; sub(/short$/, "", label) }
  if (label ~ /long$/)  { kind = "long";  sub(/long$/, "", label) }
  if (kind == "") next
  key = label " " $2 " " kind
  sum[key] += $4
  n[key]++
  if (!(label in seen)) { seen[label] = 1; order[++n_labels] = label }
}
END {
  printf "%-10s %14s %14s %14s %14s\n", "budget", "misses/token", "MiB/token", "prompt misses", "prompt MiB"
  for (i = 1; i <= n_labels; i++) {
    b = order[i]
    ok = 1
    for (what in wanted) delete wanted[what]
    split("misses mib", phases, " ")
    for (p = 1; p <= 2; p++)
      for (k = 1; k <= 2; k++) {
        key = b " " phases[p] " " (k == 1 ? "short" : "long")
        if (!(key in n)) ok = 0
      }
    if (!ok) { printf "%-10s (incomplete)\n", b; continue }
    dm = (sum[b " misses long"] / n[b " misses long"]) - (sum[b " misses short"] / n[b " misses short"])
    db = (sum[b " mib long"] / n[b " mib long"]) - (sum[b " mib short"] / n[b " mib short"])
    pm = sum[b " misses short"] / n[b " misses short"] - 8 * dm / 64
    pb = sum[b " mib short"] / n[b " mib short"] - 8 * db / 64
    printf "%-10s %14.1f %14.1f %14.0f %14.0f\n", b, dm / 64, db / 64, pm, pb
  }
}
