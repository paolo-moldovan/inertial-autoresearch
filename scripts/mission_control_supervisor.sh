#!/usr/bin/env bash

set -euo pipefail

SESSION_NAME="$(tmux display-message -p '#S' 2>/dev/null || true)"
WINDOW_INDEX="$(tmux display-message -p '#I' 2>/dev/null || echo 0)"
EXITING=0

install_tmux_bindings() {
  local session_condition
  session_condition="#{==:#{session_name},${SESSION_NAME}}"

  bind_pane_key() {
    local key="$1"
    local pane_index="$2"
    tmux bind-key -T prefix "${key}" if-shell -F "${session_condition}" \
      "select-window -t :=${WINDOW_INDEX} \\; select-pane -t ${WINDOW_INDEX}.${pane_index}" \
      "select-window -t :=${key}"
  }

  bind_pane_key 0 0
  bind_pane_key 1 1
  bind_pane_key 2 2
  bind_pane_key 3 3
  bind_pane_key 4 4

  tmux bind-key -T prefix X if-shell -F "${session_condition}" \
    "kill-session -t ${SESSION_NAME}" \
    "display-message 'Mission Control kill shortcut is only active inside the Mission Control session.'"

  tmux set-option -t "${SESSION_NAME}" status-right \
    "MC: Prefix+0..4 panes | Prefix+X kill | Prefix+q overlay"
  tmux set-option -t "${SESSION_NAME}" status-right-length 80
  tmux display-message \
    "Mission Control shortcuts: Prefix+0 control, 1 backfill, 2 monitor, 3 dashboard, 4 loop, X kill"
}

print_help() {
  cat <<'EOF'
Mission Control supervisor

Tmux shortcuts (work from any pane)
  Prefix+0         focus this supervisor pane
  Prefix+1         focus backfill pane
  Prefix+2         focus Textual monitor pane
  Prefix+3         focus web dashboard pane
  Prefix+4         focus autoresearch loop pane
  Prefix+X         kill the full Mission Control tmux session
  Prefix+q         show tmux pane numbers overlay

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

install_tmux_bindings
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
