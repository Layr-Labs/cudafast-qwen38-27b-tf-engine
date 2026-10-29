function public_log_word_char(char) {
  return char ~ /^[A-Za-z0-9._-]$/
}

function public_log_ere(value,    char, i, out) {
  out = ""
  for (i = 1; i <= length(value); i++) {
    char = substr(value, i, 1)
    if (index(".^$[](){}*+?|\\/", char)) {
      out = out "\\"
    }
    out = out char
  }
  return out
}

function public_log_replacement(value,    char, i, out) {
  out = ""
  for (i = 1; i <= length(value); i++) {
    char = substr(value, i, 1)
    if (char == "\\" || char == "&") {
      out = out "\\"
    }
    out = out char
  }
  return out
}

function public_log_add_path(value, role,    i) {
  value = value ""
  if (substr(value, 1, 1) != "/") {
    return
  }
  while (length(value) > 1 && substr(value, length(value), 1) == "/") {
    value = substr(value, 1, length(value) - 1)
  }
  if (value == "/") {
    return
  }
  for (i = 1; i <= public_log_path_count; i++) {
    if ((public_log_path[i] "") == (value "")) {
      return
    }
  }
  public_log_path_count++
  public_log_path[public_log_path_count] = value
  public_log_path_pattern[public_log_path_count] = public_log_ere(value)
  public_log_path_role[public_log_path_count] = public_log_replacement(role)
}

function public_log_add_name(value, role,    i) {
  value = value ""
  if (length(value) < 1) {
    return
  }
  for (i = 1; i <= public_log_name_count; i++) {
    if ((public_log_name[i] "") == (value "")) {
      return
    }
  }
  public_log_name_count++
  public_log_name[public_log_name_count] = value
  public_log_name_role[public_log_name_count] = role
}

function public_log_init(    i, j, pattern, role, value) {
  if (public_log_ready) {
    return
  }
  public_log_ready = 1
  public_log_add_path(ENVIRON["BENCHD_BIN_DIR"], "<benchd-dir>")
  public_log_add_path(ENVIRON["MLXFAST_TARGET_SNAPSHOT_DIR"], "<target-snapshot>")
  public_log_add_path(ENVIRON["MLXFAST_DRAFTER_SNAPSHOT_DIR"], "<drafter-snapshot>")
  public_log_add_path(ENVIRON["MLXFAST_BASELINE_WORKSPACE"], "<reference-workspace>")
  public_log_add_path(ENVIRON["MLXFAST_BASELINE_CALIBRATION"], "<calibration-file>")
  public_log_add_path(ENVIRON["PUBLIC_LOG_CALIBRATION_FILE"], "<calibration-file>")
  public_log_add_path(ENVIRON["MLXFAST_QWEN38_GOLDEN_DIR"], "<golden-dir>")
  public_log_add_path(ENVIRON["RUNNER_DIR"], "<runner-dir>")
  public_log_add_path(ENVIRON["SERVE_UP_SOCKET_DIR"], "<socket-dir>")
  public_log_add_path(ENVIRON["SANDBOX_ROOT"], "<sandbox-root>")
  public_log_add_path(ENVIRON["GITHUB_WORKSPACE"], "<workspace>")
  public_log_add_path(ENVIRON["HOME"], "<home>")
  public_log_add_name(ENVIRON["RUNNER_NAME"], "<ranked-box>")
  public_log_add_name(host, "<ranked-box>")

  for (i = 1; i < public_log_path_count; i++) {
    for (j = i + 1; j <= public_log_path_count; j++) {
      if (length(public_log_path[j]) > length(public_log_path[i])) {
        value = public_log_path[i]
        pattern = public_log_path_pattern[i]
        role = public_log_path_role[i]
        public_log_path[i] = public_log_path[j]
        public_log_path_pattern[i] = public_log_path_pattern[j]
        public_log_path_role[i] = public_log_path_role[j]
        public_log_path[j] = value
        public_log_path_pattern[j] = pattern
        public_log_path_role[j] = role
      }
    }
  }
}

function public_log_redact_name(line, value, role,    after, before, found, out, pos, start, value_length) {
  out = ""
  pos = 1
  value_length = length(value)
  while (pos <= length(line)) {
    found = index(substr(line, pos), value)
    if (!found) {
      break
    }
    start = pos + found - 1
    before = start == 1 ? "" : substr(line, start - 1, 1)
    after = substr(line, start + value_length, 1)
    if (!public_log_word_char(before) && !public_log_word_char(after) \
        && !(start >= 7 && substr(line, start - 6, 6) == "/home/")) {
      out = out substr(line, pos, start - pos) role
      pos = start + value_length
    } else {
      out = out substr(line, pos, start - pos + 1)
      pos = start + 1
    }
  }
  return out substr(line, pos)
}

function public_log_redact(line,    i) {
  public_log_init()
  for (i = 1; i <= public_log_path_count; i++) {
    gsub(public_log_path_pattern[i], public_log_path_role[i], line)
  }
  for (i = 1; i <= public_log_name_count; i++) {
    line = public_log_redact_name(line, public_log_name[i], public_log_name_role[i])
  }
  gsub(/\/home\/[A-Za-z0-9._-]+/, "<home>", line)
  return line
}

held != "" && index($0, "benchd iterate: wrote score.json") {
  print > held
  next
}

{
  public_log_omitted = length($0) - 65536
  public_log_line = public_log_omitted > 0 ? substr($0, 1, 65536) : $0
  public_log_suffix = public_log_omitted > 0 \
    ? " [" public_log_omitted " more bytes not shown]" : ""
  print public_log_redact(public_log_line) public_log_suffix
  fflush()
}
