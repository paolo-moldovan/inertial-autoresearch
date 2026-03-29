#!/usr/bin/env bash

set -euo pipefail

SESSION_NAME="$(tmux display-message -p '#S' 2>/dev/null || true)"
WINDOW_INDEX="$(tmux display-message -p '#I' 2>/dev/null || echo 0)"
EXITING=0

print_help() {
  cat <<'EOF'
Mission Control supervisor

Navigation
  0 / control      focus this supervisor pane
  1 / backfill     focus backfill pane
  2 / monitor      focus Textual monitor pane
  3 / dashboard    focus web dashboard pane
  4 / loop         focus autoresearch loop pane
  panes            show tmux pane numbers overlay
  list             list panes

Session
  kill / exit / quit / q   kill the full Mission Control tmux session
  help                     show this help

Ctrl-C in this pane also kills the whole session immediately.
EOF
}

kill_session() {
  if [[ "${EXITING}" == "1" ]]; then
    return
  fi
  EXITING=1
  if [[ -n "${SESSION_NAME}" ]]; then
    tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
  fi
}

focus_pane() {
  local pane_index="$1"
  tmux select-pane -t "${WINDOW_INDEX}.${pane_index}" 2>/dev/null || \
    echo "Pane ${pane_index} is not available."
}

trap 'echo; echo "Killing Mission Control session..."; kill_session' INT TERM
trap 'kill_session' EXIT

print_help

while true; do
  printf 'mission-control> '
  if ! IFS= read -r command; then
    echo
    exit 0
  fi

  case "${command}" in
    ""|help)
      print_help
      ;;
    0|control)
      focus_pane 0
      ;;
    1|backfill)
      focus_pane 1
      ;;
    2|monitor)
      focus_pane 2
      ;;
    3|dashboard)
      focus_pane 3
      ;;
    4|loop|autoresearch)
      focus_pane 4
      ;;
    panes)
      tmux display-panes
      ;;
    list)
      tmux list-panes -t "${WINDOW_INDEX}" -F '#P #{pane_current_command} #{pane_title}'
      ;;
    kill|exit|quit|q)
      exit 0
      ;;
    *)
      echo "Unknown command: ${command}"
      echo "Type 'help' for the available commands."
      ;;
  esac
done
