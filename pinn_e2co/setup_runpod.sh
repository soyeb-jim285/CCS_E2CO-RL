#!/usr/bin/env bash
# ==============================================================================
# RunPod Setup Script for PINN-E2CO
# Installs tmux, vim, configures them with mouse/copy support, installs deps.
#
# Expected layout after setup:
#   /workspace/
#   ├── pinn_e2co/        ← this code
#   │   ├── pinn_train.py
#   │   ├── setup_runpod.sh
#   │   └── ...
#   ├── data/             ← upload .mat files here
#   └── outputs/          ← created by training
#
# Usage (run from anywhere):
#   bash /workspace/pinn_e2co/setup_runpod.sh
# ==============================================================================
set -euo pipefail

# Always resolve paths relative to this script, not the caller's CWD
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "  PINN-E2CO RunPod Setup"
echo "  Script dir:  $SCRIPT_DIR"
echo "  Workspace:   $WORKSPACE"
echo "============================================"

# --- System packages ---
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq tmux vim xclip xsel curl git htop ncdu tree > /dev/null 2>&1
echo "  Done."

# --- Python dependencies ---
echo "[2/6] Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r "$SCRIPT_DIR/requirements.txt"
echo "  Done."

# --- Create data directory if missing ---
echo "[3/6] Ensuring directory structure..."
mkdir -p "$WORKSPACE/data"
mkdir -p "$WORKSPACE/outputs"

# Check if data files exist
DATA_OK=true
for f in states_norm_slt.mat controls_norm_slt.mat rate_norm_slt.mat TRUE_PERM_64by220.mat; do
    if [ ! -f "$WORKSPACE/data/$f" ]; then
        echo "  WARNING: $WORKSPACE/data/$f not found"
        DATA_OK=false
    fi
done
if [ "$DATA_OK" = true ]; then
    echo "  All data files present."
else
    echo "  Upload missing .mat files to $WORKSPACE/data/ before training."
fi

# --- tmux config ---
echo "[4/6] Configuring tmux..."
cat > ~/.tmux.conf << 'TMUX_EOF'
# ---- General ----
set -g default-terminal "screen-256color"
set -ga terminal-overrides ",xterm-256color:Tc"
set -g history-limit 50000
set -g base-index 1
setw -g pane-base-index 1
set -g renumber-windows on
set -g set-clipboard on
set -sg escape-time 0

# ---- Mouse support ----
set -g mouse on

# Mouse copy: select with mouse -> auto copies to tmux buffer + system clipboard
bind -T copy-mode-vi MouseDragEnd1Pane send-keys -X copy-pipe-and-cancel "xclip -selection clipboard 2>/dev/null || cat > /dev/null"
bind -T copy-mode MouseDragEnd1Pane send-keys -X copy-pipe-and-cancel "xclip -selection clipboard 2>/dev/null || cat > /dev/null"

# Middle-click paste
bind -n MouseDown2Pane paste-buffer

# Scroll wheel enters copy mode
bind -n WheelUpPane if-shell -F -t = "#{mouse_any_flag}" "send-keys -M" "if -Ft= '#{pane_in_mode}' 'send-keys -M' 'copy-mode -e'"

# ---- Vi mode ----
setw -g mode-keys vi
bind -T copy-mode-vi v send-keys -X begin-selection
bind -T copy-mode-vi y send-keys -X copy-pipe-and-cancel "xclip -selection clipboard 2>/dev/null || cat > /dev/null"
bind -T copy-mode-vi Escape send-keys -X cancel

# ---- Prefix: Ctrl-a (easier than Ctrl-b) ----
unbind C-b
set -g prefix C-a
bind C-a send-prefix

# ---- Pane splitting ----
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"
unbind '"'
unbind %

# ---- Pane navigation (Alt+arrow, no prefix needed) ----
bind -n M-Left select-pane -L
bind -n M-Right select-pane -R
bind -n M-Up select-pane -U
bind -n M-Down select-pane -D

# ---- Pane resizing (Prefix + Shift+arrow) ----
bind -r S-Left resize-pane -L 5
bind -r S-Right resize-pane -R 5
bind -r S-Up resize-pane -U 3
bind -r S-Down resize-pane -D 3

# ---- Window navigation ----
bind -n M-1 select-window -t 1
bind -n M-2 select-window -t 2
bind -n M-3 select-window -t 3
bind -n M-4 select-window -t 4
bind -n M-5 select-window -t 5

# ---- Reload config ----
bind r source-file ~/.tmux.conf \; display "Config reloaded!"

# ---- Status bar ----
set -g status-style "bg=#1a1b26,fg=#a9b1d6"
set -g status-left "#[fg=#7aa2f7,bold] #S "
set -g status-right "#[fg=#9ece6a]%H:%M #[fg=#7aa2f7]%d-%b "
set -g status-left-length 30
set -g status-right-length 40
setw -g window-status-current-style "fg=#7aa2f7,bold"
setw -g window-status-current-format " #I:#W "
setw -g window-status-format " #I:#W "

# ---- Pane borders ----
set -g pane-border-style "fg=#3b4261"
set -g pane-active-border-style "fg=#7aa2f7"
TMUX_EOF
echo "  Done."

# --- vim config ---
echo "[5/6] Configuring vim..."
cat > ~/.vimrc << 'VIM_EOF'
" ---- General ----
set nocompatible
filetype plugin indent on
syntax on

" ---- Mouse support (all modes) ----
set mouse=a
if has('mouse_sgr')
    set ttymouse=sgr
elseif !has('nvim')
    set ttymouse=xterm2
endif

" ---- Copy/paste with system clipboard ----
if has('clipboard')
    set clipboard=unnamedplus
endif
" Visual mode: yank to system clipboard
vnoremap <C-c> "+y
" Paste from system clipboard
nnoremap <C-v> "+p
inoremap <C-v> <C-r>+

" ---- Display ----
set number
set relativenumber
set cursorline
set showmatch
set showcmd
set laststatus=2
set ruler
set wildmenu
set wildmode=longest:full,full
set scrolloff=8
set sidescrolloff=8
set signcolumn=yes
set termguicolors
set background=dark

" ---- Search ----
set incsearch
set hlsearch
set ignorecase
set smartcase
" Clear search highlight with Esc
nnoremap <Esc> :nohlsearch<CR>

" ---- Indentation ----
set tabstop=4
set shiftwidth=4
set softtabstop=4
set expandtab
set autoindent
set smartindent

" ---- Python-specific ----
autocmd FileType python setlocal tabstop=4 shiftwidth=4 softtabstop=4 expandtab

" ---- Files ----
set encoding=utf-8
set fileencoding=utf-8
set nobackup
set nowritebackup
set noswapfile
set autoread
set hidden

" ---- Performance ----
set updatetime=300
set timeoutlen=500
set lazyredraw
set ttyfast

" ---- Splits ----
set splitbelow
set splitright
nnoremap <C-h> <C-w>h
nnoremap <C-j> <C-w>j
nnoremap <C-k> <C-w>k
nnoremap <C-l> <C-w>l

" ---- Quality of life ----
set backspace=indent,eol,start
set wrap
set linebreak
set noerrorbells
set novisualbell

" ---- Status line ----
set statusline=
set statusline+=\ %f              " file path
set statusline+=\ %m%r            " modified/readonly
set statusline+=%=                " right align
set statusline+=\ %y              " filetype
set statusline+=\ %l:%c           " line:col
set statusline+=\ [%p%%]          " percentage

" ---- Leader key ----
let mapleader = " "

" Quick save/quit
nnoremap <leader>w :w<CR>
nnoremap <leader>q :q<CR>
nnoremap <leader>x :x<CR>

" Buffer navigation
nnoremap <leader>bn :bnext<CR>
nnoremap <leader>bp :bprev<CR>
nnoremap <leader>bd :bdelete<CR>

" Toggle line numbers
nnoremap <leader>n :set relativenumber!<CR>

" Strip trailing whitespace
nnoremap <leader>ts :%s/\s\+$//e<CR>
VIM_EOF
echo "  Done."

# --- Verify installation ---
echo "[6/6] Verifying setup..."
echo "  Python:  $(python --version 2>&1)"
echo "  Torch:   $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "  CUDA:    $(python -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
echo "  GPU:     $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")' 2>&1)"
echo "  tmux:    $(tmux -V 2>&1)"
echo "  vim:     $(vim --version | head -1)"

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Directory layout:"
echo "    $WORKSPACE/"
echo "    ├── pinn_e2co/   (code)"
echo "    ├── data/        (.mat files)"
echo "    └── outputs/     (checkpoints, logs, plots)"
echo ""
echo "  Quick start:"
echo "    tmux new -s pinn"
echo "    cd $SCRIPT_DIR"
echo "    python pinn_train.py --epochs 5"
echo ""
echo "  Or from anywhere:"
echo "    python $SCRIPT_DIR/pinn_train.py --epochs 5"
echo ""
echo "  tmux cheatsheet:"
echo "    Prefix:       Ctrl-a"
echo "    Split horiz:  Ctrl-a |"
echo "    Split vert:   Ctrl-a -"
echo "    Navigate:     Alt+arrow"
echo "    Copy mode:    scroll up or Ctrl-a ["
echo "    Select:       v (vi mode)"
echo "    Copy:         y"
echo "    Paste:        Ctrl-a ]"
echo "    Reload conf:  Ctrl-a r"
echo "============================================"
