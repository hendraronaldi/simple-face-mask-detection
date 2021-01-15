mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableXsrfProtection = false\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml