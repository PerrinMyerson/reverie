fn main() {
    // On macOS, extension modules resolve Python symbols at load time.
    pyo3_build_config::add_extension_module_link_args();
}
